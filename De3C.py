import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class BasicBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):

    def __init__(self, in_channels=12, feature_dim=2048):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = feature_dim
        self.stem = nn.Sequential(nn.Conv1d(in_channels, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.layer1 = nn.Sequential(BasicBlock1D(64, 64), BasicBlock1D(64, 64))
        self.layer2 = nn.Sequential(BasicBlock1D(64, 128, stride=2), BasicBlock1D(128, 128))
        self.layer3 = nn.Sequential(BasicBlock1D(128, 256, stride=2), BasicBlock1D(256, 256))
        self.layer4 = nn.Sequential(BasicBlock1D(256, 512, stride=2), BasicBlock1D(512, 512))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(512, feature_dim)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f'ECG expects (B,{self.in_channels},T), got {tuple(x.shape)}')
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.projection(torch.flatten(self.avgpool(x), 1))


def build_image_encoder(pretrained=True):
    if hasattr(torchvision.models, 'ResNet50_Weights'):
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = torchvision.models.resnet50(weights=weights)
    else:
        encoder = torchvision.models.resnet50(pretrained=pretrained)
    encoder.fc = nn.Identity()
    return encoder


class De3C(nn.Module):
    def __init__(self, num_classes=3, tau_d=0.6, tau_c=0.8,
                 prior_decay=0.99, counter_weight=0.2, lambda_uni=0.5,
                 lambda_corr=0.2, feedback_threshold=0.95,
                 feedback_stable_epochs=5, feedback_weight=0.05,
                 image_pretrained=True):
        super().__init__()
        self.image_branch = build_image_encoder(image_pretrained)
        self.ecg_branch = ResNet1D()
        self.decision_branch = nn.Linear(2048, num_classes)
        self.image_classifier = nn.Linear(2048, num_classes)
        self.ecg_classifier = nn.Linear(2048, num_classes)
        self.tau_d = tau_d
        self.tau_c = tau_c
        self.prior_decay = prior_decay
        self.counter_weight = counter_weight
        self.lambda_uni = lambda_uni
        self.lambda_corr = lambda_corr
        self.feedback_threshold = feedback_threshold
        self.feedback_stable_epochs = feedback_stable_epochs
        self.feedback_weight = feedback_weight
        self.register_buffer('marginal', torch.full((num_classes,), 1.0 / num_classes))
        self.register_buffer('feedback', torch.zeros(num_classes))
        self.register_buffer('pending', torch.zeros(num_classes))
        self.register_buffer('last_target', torch.empty(0, dtype=torch.long))
        self.register_buffer('streak', torch.empty(0, dtype=torch.long))
        self.register_buffer('seen', torch.empty(0, dtype=torch.bool))
        self.epoch_active = False

    @torch.no_grad()
    def begin_epoch(self, num_samples):
        if self.epoch_active or num_samples < 1:
            raise ValueError('Invalid epoch initialization.')
        if self.last_target.numel() == 0:
            self.last_target.resize_(num_samples).fill_(-1)
            self.streak.resize_(num_samples).zero_()
            self.seen.resize_(num_samples)
        elif self.last_target.numel() != num_samples:
            raise ValueError('Sample count must remain fixed.')
        self.seen.zero_()
        self.pending.zero_()
        self.epoch_active = True

    @torch.no_grad()
    def end_epoch(self):
        if not self.epoch_active or not self.seen.all():
            raise ValueError('Each sample must be visited once per epoch.')
        mass = self.pending.sum()
        self.feedback.copy_(self.pending / mass if mass > 0 else self.pending)
        self.epoch_active = False

    @torch.no_grad()
    def ddc(self, p_f, y, pred_i, conf_i, pred_e, conf_e):
        self.marginal.mul_(self.prior_decay).add_(
            p_f.mean(0), alpha=1 - self.prior_decay)
        prior = self.marginal
        if self.feedback.sum() > 0:
            prior = (1 - self.feedback_weight) * prior + self.feedback_weight * self.feedback
        q = p_f / prior.clamp_min(1e-12)
        q = q / q.sum(1, keepdim=True).clamp_min(1e-12)
        evidence = torch.where((pred_i == pred_e) & (pred_i != y),
                               (conf_i * conf_e).sqrt(), torch.zeros_like(conf_i))
        score = q.gather(1, y[:, None]).squeeze(1) * (1 - self.counter_weight * evidence)
        return (q.argmax(1) == y) & (score >= self.tau_d)

    @torch.no_grad()
    def ccc(self, predictions, confidences, y, selected):
        targets = y.clone()
        corrected = torch.zeros_like(selected)
        reliability = confidences[0].new_zeros(y.shape)
        for a, b in ((1, 2), (0, 1), (0, 2)):
            valid = (~selected & (predictions[a] == predictions[b])
                     & (predictions[a] != y) & (confidences[a] >= self.tau_c)
                     & (confidences[b] >= self.tau_c))
            targets[valid] = predictions[a][valid]
            corrected |= valid
            weight = (confidences[a] * confidences[b]).sqrt()
            reliability = torch.maximum(reliability, weight * valid)
        return targets, corrected, reliability

    @torch.no_grad()
    def update_feedback(self, ids, targets, corrected, reliability):
        if not self.epoch_active:
            raise RuntimeError('Call begin_epoch before training.')
        ids = ids.to(device=targets.device, dtype=torch.long)
        if (ids.shape != targets.shape or (ids < 0).any()
                or (ids >= self.seen.numel()).any()
                or ids.unique().numel() != ids.numel() or self.seen[ids].any()):
            raise ValueError('Invalid or repeated sample IDs.')
        self.seen[ids] = True
        count = torch.where(self.last_target[ids] == targets,
                            self.streak[ids] + 1, torch.ones_like(targets))
        self.streak[ids] = torch.where(corrected, count, torch.zeros_like(count))
        self.last_target[ids] = torch.where(corrected, targets, torch.full_like(targets, -1))
        valid = (corrected & (self.streak[ids] >= self.feedback_stable_epochs)
                 & (reliability >= self.feedback_threshold))
        self.pending.index_add_(0, targets[valid], reliability[valid])

    def forward(self, ecg_image, ecg_signal, labels=None, epoch=0, sample_ids=None):
        h_i = torch.flatten(self.image_branch(ecg_image), 1)
        h_e = self.ecg_branch(ecg_signal)
        logits = self.decision_branch(h_i + h_e)
        if labels is None:
            return logits
        y = labels.long()
        if sample_ids is None:
            raise ValueError('Stable sample IDs are required for feedback.')
        selected = torch.ones_like(y, dtype=torch.bool)
        corrected = torch.zeros_like(selected)
        targets = y.clone()
        reliability = logits.new_zeros(y.shape)
        loss_uni = logits.new_zeros(())
        loss_corr = logits.new_zeros(())
        if epoch < 3:
            loss_sel = F.cross_entropy(logits, y)
        else:
            logit_i = self.image_classifier(h_i.detach())
            logit_e = self.ecg_classifier(h_e.detach())
            with torch.no_grad():
                p_f = logits.softmax(1)
                conf_f, pred_f = p_f.max(1)
                conf_i, pred_i = logit_i.softmax(1).max(1)
                conf_e, pred_e = logit_e.softmax(1).max(1)
                selected = self.ddc(p_f, y, pred_i, conf_i, pred_e, conf_e)
                if epoch >= 25:
                    targets, corrected, reliability = self.ccc(
                        (pred_f, pred_i, pred_e), (conf_f, conf_i, conf_e), y, selected)
            if selected.any():
                loss_sel = F.cross_entropy(logits[selected], y[selected])
                loss_uni = 0.5 * (F.cross_entropy(logit_i[selected], y[selected])
                                  + F.cross_entropy(logit_e[selected], y[selected]))
            else:
                loss_sel = logits.sum() * 0.0
                loss_uni = (logit_i.sum() + logit_e.sum()) * 0.0
            if corrected.any():
                weights = reliability[corrected]
                losses = F.cross_entropy(logits[corrected], targets[corrected], reduction='none')
                loss_corr = (weights * losses).sum() / (weights.sum() + 1e-12)
        self.update_feedback(sample_ids, targets, corrected, reliability)
        loss = loss_sel + self.lambda_uni * loss_uni + self.lambda_corr * loss_corr
        return logits, loss
