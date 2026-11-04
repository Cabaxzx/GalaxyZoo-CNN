import os
import json
import multiprocessing
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

#Caching file for global dataset statistics.
CACHE_FILE = os.path.join('data', 'dataset_stats.json')
NUM_WORKERS = 4

class ClassificationDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['type']

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def norm_params(data, batch_size=64):
    """
    Computes standard deviation and mean for dataset normalization.
    Normalization centers data around 0 with a standard deviation of 1, improving gradient descent stability.
    Results are cached to disk to omit operations on subsequent runs.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            stats = json.load(f)
        return stats['mean'], stats['std']

    trans = transforms.Compose([transforms.CenterCrop(230), transforms.ToTensor()])
    dataset = ClassificationDataset(data, transform=trans)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    channels_sum = torch.zeros(3)
    channels_sum_sqr = torch.zeros(3)
    batches = 0

    for images, _ in tqdm(loader, desc='Calculating Image Stats'):
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_sum_sqr += torch.mean(images ** 2, dim=[0, 2, 3])
        batches += 1

    mean = (channels_sum / batches).tolist()
    std = ((channels_sum_sqr / batches - torch.tensor(mean) ** 2) ** 0.5).tolist()

    with open(CACHE_FILE, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)

    return mean, std


def transformation(mean, std, img_size, training=True):
    """
    Constructs the image preprocessing pipeline.
    Training mode applies randomized spatial and color augmentations to improve model generalization.
    """
    if training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.CenterCrop(230),
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(230),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def classification_dataloader(labels_path, img_size, batch_size=64, task="classification", tolerance=None):
    """
    Initializes DataLoaders for discrete classification tasks.
    Applies stratified splitting to ensure consistent class distributions across subsets.
    """
    labels = pd.read_csv(labels_path)
    class_probs = labels[['Class1.1', 'Class1.2', 'Class1.3']].values
    labels['type'] = class_probs.argmax(axis=1)

    if tolerance is not None:
        labels['confidence'] = class_probs.max(axis=1)
        labels_filtered = labels[labels['confidence'] >= tolerance].copy()
    else:
        labels_filtered = labels.copy()

    if task == "binary":
        labels_filtered = labels_filtered[labels_filtered['type'] != 2].copy()

    labels_filtered['filepath'] = labels_filtered['GalaxyID'].apply(lambda x: os.path.join('data/images', f"{x}.jpg"))
    data = labels_filtered[['filepath', 'type']]

    #Stratified split enforces proportional representation of minority classes in validation and test sets.
    train_data, temp_split = train_test_split(data, test_size=0.3, random_state=42, stratify=data['type'])
    val_data, test_data = train_test_split(temp_split, test_size=(1 / 3), random_state=42, stratify=temp_split['type'])

    mean, std = norm_params(data)
    train_trans = transformation(mean, std, img_size, training=True)
    val_trans = transformation(mean, std, img_size, training=False)

    train_loader = DataLoader(ClassificationDataset(train_data, train_trans), batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(ClassificationDataset(val_data, val_trans), batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(ClassificationDataset(test_data, val_trans), batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, train_data


class RegressionDataset(Dataset):
    """
    Dataset handler for continuous fractional targets (Task 2: Q1, Q2, Q7).
    Implements target softening and Bayesian conditional probability extraction.
    """

    def __init__(self, dataframe, target_cols, transform, architecture="flat", is_training=False):
        self.dataframe = dataframe
        self.transform = transform
        self.target_cols = target_cols
        self.architecture = architecture
        self.is_training = is_training

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        row = self.dataframe.iloc[idx]

        if self.architecture == "tree":
            r11, r12, r13 = row['Class1.1'], row['Class1.2'], row['Class1.3']
            r21, r22 = row['Class2.1'], row['Class2.2']
            r71, r72, r73 = row['Class7.1'], row['Class7.2'], row['Class7.3']
            r61, r62 = row['Class6.1'], row['Class6.2']
            r8_raw = np.array([row[f'Class8.{i}'] for i in range(1, 8)], dtype=np.float32)

            if self.is_training:
                #Extracts underlying conditional probabilities from marginal crowdsourced votes.
                sum_q2 = r21 + r22
                c21, c22 = (r21 / sum_q2, r22 / sum_q2) if sum_q2 > 1e-6 else (0.5, 0.5)

                sum_q7 = r71 + r72 + r73
                c71, c72, c73 = (r71 / sum_q7, r72 / sum_q7, r73 / sum_q7) if sum_q7 > 1e-6 else (1 / 3, 1 / 3, 1 / 3)
                sum_q6 = r61 + r62
                c61, c62 = (r61 / sum_q6, r62 / sum_q6) if sum_q6 > 1e-6 else (0.5, 0.5)
                sum_q8 = r8_raw.sum()
                c8 = r8_raw / sum_q8 if sum_q8 > 1e-6 else np.ones(7, dtype=np.float32) / 7.0

                #Target Softening: Compresses [0.0, 1.0] bounds to [0.01, 0.99]
                s11, s12, s13 = r11 * 0.97 + 0.01, r12 * 0.97 + 0.01, r13 * 0.97 + 0.01
                s21, s22 = c21 * 0.98 + 0.01, c22 * 0.98 + 0.01
                s71, s72, s73 = c71 * 0.97 + 0.01, c72 * 0.97 + 0.01, c73 * 0.97 + 0.01
                s61, s62 = c61 * 0.98 + 0.01, c62 * 0.98 + 0.01
                s8 = c8 * 0.93 + 0.01

                #Reconstructs the target vectors by multiplying the softened conditionals back down the tree hierarchy.
                f11, f12, f13 = s11, s12, s13
                f21, f22 = s21 * s12, s22 * s12
                f71, f72, f73 = s71 * s11, s72 * s11, s73 * s11
                f61, f62 = s61, s62
                f8 = s8 * f61
                targets = np.concatenate([[f11, f12, f13, f21, f22, f71, f72, f73, f61, f62], f8])

            else:
                targets = np.concatenate([[r11, r12, r13, r21, r22, r71, r72, r73, r61, r62], r8_raw])

        else:
            targets = row[self.target_cols].values.astype(np.float32)
            if self.is_training:
                #Standard bound softening applied to independent target arrays.
                targets = (targets * 0.98) + 0.01

        return image, torch.tensor(targets, dtype=torch.float32)


def regression_dataloaders(labels_path, img_size, batch_size=64, architecture="flat"):
    """
    Initializes DataLoaders for continuous fractional regression tasks.
    Configured specifically for Task 2 (8 outputs).
    """
    labels = pd.read_csv(labels_path)
    labels['filepath'] = labels['GalaxyID'].apply(lambda x: os.path.join('data/images', f"{x}.jpg"))

    #Derives discrete labels strictly for maintaining structural parity during splitting.
    labels['type'] = labels[['Class1.1', 'Class1.2', 'Class1.3']].values.argmax(axis=1)

    all_possible_cols = [
        'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class7.1', 'Class7.2', 'Class7.3',
        'Class6.1', 'Class6.2', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7'
    ]

    data = labels[['filepath', 'type'] + all_possible_cols]

    train_data, temp_split = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_split, test_size=(1 / 3), random_state=42)

    mean, std = norm_params(data)
    train_trans = transformation(mean, std, img_size, training=True)
    val_trans = transformation(mean, std, img_size, training=False)

    if architecture == "tree":
        sampler = None
        shuffle_train = True
    else:
        print("Calculating Inverse Frequency Weights for Flat Architecture...")
        train_targets = train_data[all_possible_cols].values
        col_sums = train_targets.sum(axis=0)
        col_weights = 1.0 / (col_sums + 1.0)
        sample_weights = np.dot(train_targets, col_weights) + 1.0
        sampler = WeightedRandomSampler(
             weights=torch.DoubleTensor(sample_weights),
             num_samples=len(sample_weights),
             replacement=True
         )
        shuffle_train = False

    train_loader = DataLoader(
        RegressionDataset(train_data, all_possible_cols, train_trans, architecture, is_training=True),
        batch_size=batch_size, shuffle=shuffle_train, sampler=sampler, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True)

    val_loader = DataLoader(
        RegressionDataset(val_data, all_possible_cols, val_trans, architecture, is_training=False),
        batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True)

    test_loader = DataLoader(
        RegressionDataset(test_data, all_possible_cols, val_trans, architecture, is_training=False),
        batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True)

    return train_loader, val_loader, test_loader

