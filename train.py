import torch
import wandb
from tqdm import tqdm
from losses import CBMLoss
from typing import Dict

class Trainer:
    """
    Manages the training and validation of the CBM framework, integrating with Weights & Biases.
    """
    def __init__(self, model: torch.nn.Module, criterion: CBMLoss, optimizer: torch.optim.Optimizer, device: torch.device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = {
            "loss/total": 0.0,
            "loss/task_classification": 0.0,
            "loss/concept_fidelity": 0.0,
            "loss/leakage_entropy": 0.0,
            "loss/leakage_ortho": 0.0
        }
        
        for images, concepts, labels in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            concepts = concepts.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            y_hat, c_hat = self.model(images)
            
            loss, metrics = self.criterion(y_hat, labels, c_hat, concepts)
            
            loss.backward()
            self.optimizer.step()
            
            for k, v in metrics.items():
                epoch_metrics[k] += v
                
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
            
        return epoch_metrics

    def validate_epoch(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        task_correct = 0
        concept_correct = 0
        total_samples = 0
        total_concepts = 0
        
        with torch.no_grad():
            for images, concepts, labels in tqdm(dataloader, desc="Validation", leave=False):
                images = images.to(self.device)
                concepts = concepts.to(self.device)
                labels = labels.to(self.device)
                
                y_hat, c_hat = self.model(images)
                
                # Task Accuracy
                preds = torch.argmax(y_hat, dim=1)
                task_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
                # Concept Accuracy (Threshold = 0.5)
                c_preds = (c_hat >= 0.5).float()
                concept_correct += (c_preds == concepts).sum().item()
                total_concepts += concepts.numel()
                
        metrics = {
            "val/task_accuracy": task_correct / total_samples,
            "val/concept_accuracy": concept_correct / total_concepts
        }
        return metrics

    def fit(self, train_loader, val_loader, epochs: int = 10, project_name: str = "cbm-leakage-mitigation"):
        wandb.init(project=project_name, config={"epochs": epochs})
        wandb.config.update({"model": self.model.__class__.__name__})
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            # Log to wandb
            wandb.log({**train_metrics, **val_metrics})
            
            print(f"Train Loss: {train_metrics['loss/total']:.4f} | Val Task Acc: {val_metrics['val/task_accuracy']:.4f} | Val Concept Acc: {val_metrics['val/concept_accuracy']:.4f}")
            
        wandb.finish()
