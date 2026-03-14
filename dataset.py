import torch
from torch.utils.data import Dataset
import numpy as np

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

class MockBiologicalDataset(BaseConceptDataset):
    """
    Mock dataset for validating the CBM pipeline.
    Simulates biological images (e.g., leaves) with binary concepts 
    (e.g., 'halo', 'sporulation', 'angular lesion') and logical classes.
    """
    def __init__(self, num_samples: int = 100, num_concepts: int = 3, image_size: tuple = (3, 224, 224), transform=None):
        super().__init__(transform)
        self.num_samples = num_samples
        self.num_concepts = num_concepts
        self.image_size = image_size
        
        # Randomly generate binary concepts
        self.concepts = torch.randint(0, 2, (num_samples, num_concepts), dtype=torch.float32)
        
        # Rule-based class derivation:
        # e.g., if sum(concepts) >= 2, it's disease A (class 1), else Healthy/disease B (class 0)
        self.labels = (self.concepts.sum(dim=1) >= (num_concepts / 2.0)).long()
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Generate a random image tensor
        image = torch.randn(self.image_size)
        if self.transform:
            image = self.transform(image)
            
        concept_vector = self.concepts[idx]
        class_label = self.labels[idx].item()
        
        return image, concept_vector, class_label

def generate_mock_dataloaders(batch_size: int = 16):
    """
    Helper function to generate train and validation dataloaders for the mock dataset.
    """
    from torch.utils.data import DataLoader
    
    train_dataset = MockBiologicalDataset(num_samples=800)
    val_dataset = MockBiologicalDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
