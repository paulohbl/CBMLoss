import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BaseConceptDataset(Dataset):
    """
    Abstract base class for Concept Datasets.
    Returns (image, concept_vector, class_label).
    """
    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError

class DiskConceptDataset(BaseConceptDataset):
    """
    Generic dataset loader for CSV-based concept datasets.
    """
    def __init__(self, data_dir: str, csv_name: str, images_dir: str = "images", transform=None):
        super().__init__(transform)
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.csv_path = os.path.join(data_dir, csv_name)
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}. Please run download_datasets.py first!")
            
        self.df = pd.read_csv(self.csv_path)
        
        # All columns starting with 'concept_' are our concept binary vectors
        self.concept_cols = [c for c in self.df.columns if c.startswith('concept_')]
        self.num_concepts = len(self.concept_cols)
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        img_name = str(row['image_name'])
        img_path = os.path.join(self.data_dir, self.images_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Extract concepts dynamically
        concept_vector = torch.tensor([row[c] for c in self.concept_cols], dtype=torch.float32)
        class_label = int(row['label'])
        
        return image, concept_vector, class_label

class MockBiologicalDataset(BaseConceptDataset):
    """
    Fast mock generator (Tensors in memory) for quick debugging.
    """
    def __init__(self, num_samples: int = 100, num_concepts: int = 3, image_size: tuple = (3, 224, 224), transform=None):
        super().__init__(transform)
        self.num_samples = num_samples
        self.num_concepts = num_concepts
        self.image_size = image_size
        self.concepts = torch.randint(0, 2, (num_samples, num_concepts), dtype=torch.float32)
        self.labels = (self.concepts.sum(dim=1) >= (num_concepts / 2.0)).long()
        
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        image = torch.randn(self.image_size)
        return image, self.concepts[idx], self.labels[idx].item()

def get_dataloaders(dataset_name: str, batch_size: int = 32):
    """
    Returns train_loader, val_loader, num_concepts, num_classes
    """
    # Base transforms for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == "mock":
        train_ds = MockBiologicalDataset(num_samples=800, num_concepts=3, transform=None)
        val_ds = MockBiologicalDataset(num_samples=200, num_concepts=3, transform=None)
        num_concepts = 3
        num_classes = 2
        
    elif dataset_name == "synthetic_leaf":
        data_dir = "data/synthetic_leaf"
        train_ds = DiskConceptDataset(data_dir, "train.csv", transform=transform)
        val_ds = DiskConceptDataset(data_dir, "val.csv", transform=transform)
        num_concepts = train_ds.num_concepts
        num_classes = 2
        
    elif dataset_name == "cub200":
        data_dir = "data/CUB_200_2011"
        train_ds = DiskConceptDataset(data_dir, "train.csv", transform=transform)
        val_ds = DiskConceptDataset(data_dir, "val.csv", transform=transform)
        num_concepts = train_ds.num_concepts
        num_classes = 200 # CUB has 200 bird species
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, num_concepts, num_classes
