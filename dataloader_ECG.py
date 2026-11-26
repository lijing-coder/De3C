from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dependency_ECG import *
from utils_ECG import (
    encode_meta_label, load_ecg_array, load_image_array,
    prepare_mm_ecg_metadata, resolve_csv_path, resolve_npy_dir, save_split_indices,
    split_by_patient, verify_mm_ecg_files,
)


class MMECGDataset(Dataset):
    def __init__(self, image_dir, img_info, file_list, shape=(224, 224),
                 is_test=False, num_class=num_label, return_index=False,
                 image_normalization='auto', image_npy_dir=None, ecg_npy_dir=None):
        self.image_dir = str(image_dir)
        self.img_info = img_info
        df = prepare_mm_ecg_metadata(img_info, image_dir)
        image_folder = resolve_npy_dir(
            image_dir, 'image_npy', [Path(p).name for p in df['_image_path']],
            IMAGE_NPY_DIR if image_npy_dir is None else image_npy_dir,
        )
        ecg_folder = resolve_npy_dir(
            image_dir, 'ecg_npy', [Path(p).name for p in df['_ecg_path']],
            ECG_NPY_DIR if ecg_npy_dir is None else ecg_npy_dir,
        )
        df['_image_path'] = [str(image_folder / Path(p).name) for p in df['_image_path']]
        df['_ecg_path'] = [str(ecg_folder / Path(p).name) for p in df['_ecg_path']]
        self.total_img_info = df
        self.file_list = [int(i) for i in file_list]
        if len(self.file_list) != len(set(self.file_list)):
            raise ValueError('Duplicate CSV indices in this split.')
        if any(i < 0 or i >= len(self.total_img_info) for i in self.file_list):
            raise IndexError('Split indices must be CSV row positions.')
        self.shape = None if shape is None else tuple(int(x) for x in shape)
        if self.shape is not None and (len(self.shape) != 2 or min(self.shape) <= 0):
            raise ValueError('shape must be (height, width) with positive dimensions, or None.')
        self.is_test = is_test
        self.num_class = num_label
        self.return_index = return_index
        self.image_normalization = image_normalization
        part = self.total_img_info.iloc[self.file_list]
        self.targets = part['_target'].to_numpy(dtype=np.int64, copy=True)
        self.sample_ids = part['id'].astype(str).tolist()
        self.patient_ids = part['_patient_id'].astype(str).tolist()

    def get_diagnosis_labels(self):

        return torch.from_numpy(self.targets.copy()).long()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.total_img_info.iloc[self.file_list[index]]
        image = torch.from_numpy(load_image_array(row['_image_path'], self.image_normalization))
        if self.shape is not None and tuple(image.shape[-2:]) != self.shape:
            image = F.interpolate(
                image.unsqueeze(0), size=self.shape, mode='bilinear', align_corners=False,
            ).squeeze(0)
        ecg = torch.from_numpy(load_ecg_array(row['_ecg_path']))

        metadata = torch.from_numpy(encode_meta_label())
        labels = [torch.tensor(self.targets[index], dtype=torch.long)]
        batch = (image, ecg, metadata, labels)
        if self.return_index:

            return batch + (int(index),)
        return batch


SkinDataset = MMECGDataset


def generate_dataloader(shape, batch_size, num_workers, data_mode,
                        return_train_index=False, *, data_root=None,
                        csv_path=None, split_seed=SPLIT_SEED,
                        shuffle_train=False, verify_files=True,
                        image_normalization='auto', save_split=True,
                        image_npy_dir=None, ecg_npy_dir=None):


    root = Path(source_dir if data_root is None else data_root).expanduser()
    csv_file = resolve_csv_path(root, csv_path)
    df = prepare_mm_ecg_metadata(pd.read_csv(csv_file), root)
    image_folder = resolve_npy_dir(
        root, 'image_npy', [Path(p).name for p in df['_image_path']],
        IMAGE_NPY_DIR if image_npy_dir is None else image_npy_dir,
    )
    ecg_folder = resolve_npy_dir(
        root, 'ecg_npy', [Path(p).name for p in df['_ecg_path']],
        ECG_NPY_DIR if ecg_npy_dir is None else ecg_npy_dir,
    )
    df['_image_path'] = [str(image_folder / Path(p).name) for p in df['_image_path']]
    df['_ecg_path'] = [str(ecg_folder / Path(p).name) for p in df['_ecg_path']]
    print(f'MM-ECG CSV: {csv_file}', flush=True)
    print(f'MM-ECG image directory: {image_folder}', flush=True)
    print(f'MM-ECG signal directory: {ecg_folder}', flush=True)
    train_indices, test_indices = split_by_patient(df, seed=split_seed)
    if verify_files:
        verify_mm_ecg_files(df)
    if save_split:
        save_split_indices(df, train_indices, test_indices, root / 'splits', split_seed)

    train_dataset = MMECGDataset(
        image_dir=root, img_info=df, file_list=train_indices, shape=shape,
        is_test=False, return_index=return_train_index,
        image_normalization=image_normalization,
        image_npy_dir=image_folder, ecg_npy_dir=ecg_folder,
    )
    test_dataset = MMECGDataset(
        image_dir=root, img_info=df, file_list=test_indices, shape=shape,
        is_test=True, return_index=False,
        image_normalization=image_normalization,
        image_npy_dir=image_folder, ecg_npy_dir=ecg_folder,
    )
    if set(train_dataset.patient_ids) & set(test_dataset.patient_ids):
        raise RuntimeError('Patient overlap between train and test.')
    train_generator = torch.Generator().manual_seed(int(split_seed))
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), drop_last=False)
    train_loader = DataLoader(train_dataset, shuffle=shuffle_train,
                              generator=train_generator, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    print(f'MM-ECG | train={len(train_dataset)} | test={len(test_dataset)} '
          f'| patient overlap=0 | split_seed={split_seed}')
    for label, name in enumerate(CLASS_NAMES):
        print(f'  {label}: {name} | train={int((train_dataset.targets == label).sum())}'
              f' | test={int((test_dataset.targets == label).sum())}')
    return train_loader, test_loader
