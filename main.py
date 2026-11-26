import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from De3C import De3C
from dataloader_ECG import generate_dataloader
from dependency_ECG import source_dir, num_label
from utils_ECG import set_seed, add_noisy_label, calculate_metrics


def train(model, loader, optimizer, device, epoch, noise_rate, seed):
    model.train()
    model.begin_epoch(len(loader.dataset))
    total_loss = 0.0
    total = 0
    for image, signal, _, labels, indices in loader:
        image = image.to(device)
        signal = signal.to(device)
        labels = labels[0].long().to(device)
        indices = indices.long().to(device)
        noisy_labels = add_noisy_label(labels, noise_rate, num_label, seed)
        optimizer.zero_grad()
        _, loss = model(image, signal, noisy_labels, epoch=epoch, sample_ids=indices)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total += labels.numel()
    model.end_epoch()
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    targets, predictions = [], []
    for image, signal, _, labels in loader:
        logits = model(image.to(device), signal.to(device))
        targets.extend(labels[0].tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return calculate_metrics(targets, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=source_dir)
    parser.add_argument('--csv-path', default=None)
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--ecg-dir', default=None)
    parser.add_argument('--noise-rate', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--aux-lr', type=float, default=1e-4)
    parser.add_argument('--tau-d', type=float, default=0.6)
    parser.add_argument('--tau-c', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.noise_rate <= 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error('Invalid noise rate, epochs or batch size.')
    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader = generate_dataloader(
        shape=(args.image_size, args.image_size), batch_size=args.batch_size,
        num_workers=args.num_workers, data_mode='Normal', return_train_index=True,
        data_root=args.data_root, csv_path=args.csv_path, split_seed=args.seed,
        image_npy_dir=args.image_dir, ecg_npy_dir=args.ecg_dir, save_split=False,
    )
    model = De3C(num_classes=num_label, tau_d=args.tau_d, tau_c=args.tau_c,
                  image_pretrained=not args.no_pretrained).to(device)
    heads = (list(model.decision_branch.parameters()) +
             list(model.image_classifier.parameters()) +
             list(model.ecg_classifier.parameters()))
    optimizer = Adam([
        {'params': model.image_branch.parameters(), 'lr': args.lr},
        {'params': model.ecg_branch.parameters(), 'lr': args.aux_lr},
        {'params': heads, 'lr': args.aux_lr},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, device, epoch,
                     args.noise_rate, args.seed)
        scores = evaluate(model, test_loader, device)
        print('Epoch {:03d} | Loss {:.4f} | F1 {:.4f} | Kappa {:.4f} | Acc {:.4f}'.format(
            epoch + 1, loss, scores['f1'], scores['kappa'], scores['accuracy']))
        scheduler.step()


if __name__ == '__main__':
    main()
