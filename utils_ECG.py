import hashlib

import json

import math

import os

from pathlib import Path

import re

import numpy as np

import pandas as pd

from dependency_ECG import *

def _layout_roots(data_root):
    root = Path(data_root).expanduser()
    anchors = [root]

    if root.name == 'dataset':
        anchors.append(root.parent)
    roots = []
    for anchor in anchors:
        for relative in ('', 'dataset', 'MM-ECG', 'MM_ECG',
                         'dataset/MM-ECG', 'dataset/MM_ECG'):
            candidate = anchor / relative
            if candidate not in roots:
                roots.append(candidate)
    return roots

def resolve_npy_dir(data_root, modality, filenames, explicit_dir=None):


    if explicit_dir is not None:
        folder = Path(explicit_dir).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f'{modality} directory does not exist: {folder}')
        return folder.resolve()
    if modality not in ('image_npy', 'ecg_npy'):
        raise ValueError(f'Unknown modality: {modality}')
    expected = set(filenames)
    candidates = []
    for root in _layout_roots(data_root):
        for candidate in (root / modality,
                          root / modality / 'MM-ECG' / modality,
                          root / modality / 'MM_ECG' / modality):
            if candidate not in candidates:
                candidates.append(candidate)
    matches = {}
    for folder in candidates:
        if folder.is_dir():

            found_names = {p.name for p in folder.iterdir() if p.suffix == '.npy'}
            count = len(expected & found_names)
            if count:
                matches[folder.resolve()] = count
    if not matches:
        checked = '\n'.join(str(p) for p in candidates[:9])
        argument = 'image_npy_dir' if modality == 'image_npy' else 'ecg_npy_dir'
        raise FileNotFoundError(
            f'No {modality} folder containing CSV-matching .npy filenames was found.\n'
            f'Checked locations include:\n{checked}\n'
            f'Pass {argument}="/actual/folder/containing/npy/files" explicitly.'
        )
    best_count = max(matches.values())
    best = [folder for folder, count in matches.items() if count == best_count]
    if len(best) != 1:
        raise ValueError(
            f'Multiple matching {modality} folders; specify its directory explicitly:\n'
            + '\n'.join(str(folder) for folder in best)
        )
    return best[0]

def resolve_csv_path(data_root, csv_path=None):
    if csv_path is not None:
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'MM-ECG CSV not found: {path}')
        return path
    names = ('final_report_with_mapping_with_ecg_new.csv',
             'final_report_with_mapping_with_ecg_new(1).csv')
    root = Path(data_root).expanduser()
    for layout_root in _layout_roots(root):
        for folder in (layout_root, layout_root / 'meta'):
            for name in names:
                if (folder / name).is_file():
                    return folder / name
    raise FileNotFoundError(
        f'Put {names[0]} in {root}, or pass csv_path="/path/to/your.csv".'
    )

def diagnosis_label_from_row(row):
    mapped = str(row['answer_text_mapped']).strip().upper()
    if mapped not in LABEL_TO_INDEX:
        raise ValueError(f'Unknown answer_text_mapped: {mapped!r}')
    label = LABEL_TO_INDEX[mapped]
    if 'answer_text' in row and str(row['answer_text']).strip() != CLASS_NAMES[label]:
        raise ValueError(
            f'Inconsistent label for {row.get("id", "unknown")}: '
            f'{mapped} should mean {CLASS_NAMES[label]!r}, '
            f'got {row["answer_text"]!r}.'
        )
    return label

def _patient_id(row):
    for column in ('patient_id', 'subject_id'):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    for column in ('image', 'ecg_path'):
        match = re.search(r'(?:^|/)(p\d+)/s\d+(?:/|$)', str(row.get(column, '')))
        if match:
            return match.group(1)
    raise ValueError(f'Cannot determine patient ID for sample {row.get("id", "unknown")}')

def prepare_mm_ecg_metadata(img_info, data_root, image_npy_dir=None, ecg_npy_dir=None):

    df = img_info.reset_index(drop=True).copy()
    required = {'id', 'image', 'answer_text_mapped'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('MM-ECG CSV is empty.')
    if df['id'].isna().any() or df['id'].astype(str).duplicated().any():
        raise ValueError('MM-ECG sample IDs must be present and unique.')
    root = Path(data_root).expanduser()
    image_folder = Path(image_npy_dir) if image_npy_dir is not None else root / 'image_npy'
    ecg_folder = Path(ecg_npy_dir) if ecg_npy_dir is not None else root / 'ecg_npy'
    image_paths, ecg_paths, targets, patients = [], [], [], []
    for _, row in df.iterrows():
        if pd.isna(row['image']) or not str(row['image']).strip():
            raise ValueError(f'Missing image filename for {row["id"]}')
        image_name = Path(str(row['image']).strip()).with_suffix('.npy').name
        ecg_value = row.get('ecg_npy_new')
        if pd.isna(ecg_value) or not str(ecg_value).strip():
            ecg_value = row.get('ecg_path')
        if pd.isna(ecg_value) or not str(ecg_value).strip():
            raise ValueError(f'Missing ECG filename for {row["id"]}')
        ecg_name = Path(str(ecg_value).strip()).name
        if not ecg_name.endswith('.npy'):
            raise ValueError(f'Expected ECG .npy filename, got {ecg_name!r}')
        image_paths.append(str(image_folder / image_name))
        ecg_paths.append(str(ecg_folder / ecg_name))
        targets.append(diagnosis_label_from_row(row))
        patients.append(_patient_id(row))
    df['_image_path'] = image_paths
    df['_ecg_path'] = ecg_paths
    df['_target'] = np.asarray(targets, dtype=np.int64)
    df['_patient_id'] = patients

    for column in ('_image_path', '_ecg_path'):
        if df[column].duplicated().any():
            raise ValueError(f'Duplicate recording filenames in {column}.')
    return df

def split_by_patient(df, test_fraction=TEST_FRACTION, seed=SPLIT_SEED,
                     candidates=SPLIT_CANDIDATES):


    if not 0 < test_fraction < 1 or candidates < 1:
        raise ValueError('Require 0 < test_fraction < 1 and candidates >= 1.')
    groups = (df.groupby('_patient_id', sort=True)['_target'].value_counts()
              .unstack(fill_value=0).reindex(columns=range(num_label), fill_value=0))
    if len(groups) < 2:
        raise ValueError('At least two patients are required for train/test.')
    counts = groups.to_numpy(dtype=np.int64)
    totals = counts.sum(axis=0)
    if (totals == 0).any():
        raise ValueError('The CSV must contain all three MM-ECG classes.')
    sizes = counts.sum(axis=1)
    target_n = int(math.ceil(len(df) * test_fraction))
    target_counts = totals * (target_n / len(df))
    rng = np.random.default_rng(seed)
    best_score, best_groups = None, None
    for _ in range(candidates):
        order = rng.permutation(len(groups))
        cumulative = np.cumsum(sizes[order])
        position = int(np.searchsorted(cumulative, target_n))
        for end in (position, position + 1):
            if not 1 <= end < len(groups):
                continue
            selected = counts[order[:end]].sum(axis=0)
            if (selected == 0).any() or (selected >= totals).any():
                continue
            score = (
                abs(int(selected.sum()) - target_n),
                float(np.sum((selected - target_counts) ** 2 / np.maximum(target_counts, 1))),
            )
            if best_score is None or score < best_score:
                best_score, best_groups = score, order[:end].copy()
    if best_groups is None:
        raise ValueError('Cannot create patient-disjoint train/test with every class in both sets.')
    in_test = df['_patient_id'].isin(groups.index[best_groups]).to_numpy()

    return np.flatnonzero(~in_test), np.flatnonzero(in_test)

def save_split_indices(df, train_indices, test_indices, directory, seed):

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, indices in (('train', train_indices), ('test', test_indices)):
        part = df.iloc[indices]
        pd.DataFrame({
            'indexes': indices,
            'id': part['id'].to_numpy(),
            'patient_id': part['_patient_id'].to_numpy(),
            'label': part['_target'].to_numpy(),
        }).to_csv(directory / f'{name}_indexes.csv', index=False)
    signature = df[['id', '_patient_id', '_target']].to_csv(index=False).encode('utf-8')
    info = {
        'seed': int(seed), 'method': 'patient_grouped_class_balanced_holdout',
        'train_samples': len(train_indices), 'test_samples': len(test_indices),
        'label_to_index': LABEL_TO_INDEX, 'class_names': CLASS_NAMES,
        'csv_order_and_labels_sha256': hashlib.sha256(signature).hexdigest(),
    }
    (directory / 'split_info.json').write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )

def verify_mm_ecg_files(df):
    details = []
    total_missing = 0
    for name, column in (('image_npy', '_image_path'), ('ecg_npy', '_ecg_path')):
        missing = [path for path in df[column] if not Path(path).is_file()]
        total_missing += len(missing)
        if missing:
            details.append(f'{name}: {len(missing)} missing; examples:\n' + '\n'.join(missing[:4]))
    if total_missing:
        raise FileNotFoundError(
            f'{total_missing} MM-ECG files are missing.\n' + '\n'.join(details)
            + '\nCheck the image and ECG directories printed above. '
            'Samples were not silently dropped.'
        )

def load_image_array(path, normalization='auto'):


    if normalization not in ('auto', 'none', '255'):
        raise ValueError("image_normalization must be 'auto', 'none', or '255'.")
    image = np.load(path, allow_pickle=False)
    if image.ndim != 3:
        raise ValueError(f'Expected 3D RGB image at {path}, got {image.shape}')
    if image.shape[0] == 3:
        pass
    elif image.shape[-1] == 3:
        image = image.transpose(2, 0, 1)
    else:
        raise ValueError(f'Expected 3 RGB channels at {path}, got {image.shape}')
    integer_input = np.issubdtype(image.dtype, np.integer)
    image = np.asarray(image, dtype=np.float32)
    if not image.size or not np.isfinite(image).all():
        raise ValueError(f'Empty image or NaN/Inf values at {path}')
    low, high = float(image.min()), float(image.max())
    if normalization == '255':
        image = image / 255.0
    elif normalization == 'auto':
        if integer_input:
            if low < 0 or high > 255:
                raise ValueError(f'Integer image range [{low}, {high}] is not [0,255]: {path}')
            image = image / 255.0
        elif low >= 0 and high > 1:
            if high > 255:
                raise ValueError(f'Unknown image scale at {path}; set image_normalization explicitly.')
            image = image / 255.0
    return np.ascontiguousarray(image, dtype=np.float32)

def load_ecg_array(path):

    ecg = np.load(path, allow_pickle=False)
    if ecg.shape == ECG_SHAPE[::-1]:
        ecg = ecg.T
    if ecg.shape != ECG_SHAPE:
        raise ValueError(f'Expected ECG shape {ECG_SHAPE} at {path}, got {ecg.shape}')
    ecg = np.ascontiguousarray(ecg, dtype=np.float32)
    if not np.isfinite(ecg).all():
        raise ValueError(f'ECG contains NaN/Inf values: {path}')
    return ecg

def encode_meta_label(img_info=None, index_num=None):

    return np.zeros(META_DIM, dtype=np.float32)

def set_seed(seed=15):
    import random
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def add_noisy_label(labels, noise_rate, num_classes, seed=42):
    import torch
    if not 0 <= noise_rate <= 1 or num_classes < 2:
        raise ValueError('Invalid noise rate or class count.')
    noisy = labels.clone()
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    count = int(labels.numel() * noise_rate)
    indices = torch.randperm(labels.numel(), generator=generator, device=labels.device)[:count]
    replacements = torch.randint(0, num_classes - 1, (count,), generator=generator, device=labels.device)
    noisy[indices] = replacements + (replacements >= labels[indices]).long()
    return noisy


def calculate_metrics(targets, predictions):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    return {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(targets, predictions),
    }
