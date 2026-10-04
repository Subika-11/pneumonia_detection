import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms

class RSNADataset(Dataset):
    """
    PyTorch Dataset for RSNA Pneumonia radiographs.
    Loads standardized 224x224 grayscale images, expands them to 3 channels,
    and applies data transforms and ImageNet normalization.
    """
    def __init__(self, df, images_dir=r"D:\rsna_pneumonia\images_224", transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = row['study_uid']
        target = float(row['target'])
        
        # Load image from cache
        img_path = os.path.join(self.images_dir, f"{study_uid}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found in cache: {img_path}")
            
        # Open image as grayscale and convert to RGB (3 identical channels)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        label = torch.tensor(target, dtype=torch.float32)
        
        metadata = {
            'study_uid': study_uid,
            'patient_id': row.get('patient_id', ''),
            'patient_age': row.get('patient_age', -1),
            'patient_sex': row.get('patient_sex', 'Unknown'),
            'view_position': row.get('view_position', 'Unknown')
        }
        
        return img, label, metadata

def get_transforms():
    """
    Returns train and evaluation transforms.
    Medical CXRs require subtle augmentations that preserve anatomical validity:
    - Horizontal flip (p=0.5): valid for symmetric chest anatomy
    - Slight rotation (+-10 deg): simulates patient positioning variation
    - Slight affine scaling (0.95 to 1.05)
    - Color jitter (+-10% brightness/contrast): simulates varying exposure
    - Normalization: ImageNet mean & standard deviation expected by DenseNet121
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(splits_csv=r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\data\processed\splits.csv",
                    images_dir=r"D:\rsna_pneumonia\images_224",
                    batch_size=32,
                    num_workers=4):
    """
    Loads splits.csv and creates reproducible DataLoaders for Train, Val, and Test.
    """
    df = pd.read_csv(splits_csv)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    train_tf, val_tf = get_transforms()
    
    train_dataset = RSNADataset(train_df, images_dir=images_dir, transform=train_tf)
    val_dataset = RSNADataset(val_df, images_dir=images_dir, transform=val_tf)
    test_dataset = RSNADataset(test_df, images_dir=images_dir, transform=val_tf)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
