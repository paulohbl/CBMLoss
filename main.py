import torch
import torch.optim as optim
from dataset import generate_mock_dataloaders
from model import ConceptBottleneckModel
from losses import CBMLoss
from train import Trainer
from evaluate import evaluate_concept_intervention
import wandb
import os

def main():
    # Set wandb offline to avoid hanging without login
    os.environ["WANDB_MODE"] = "offline"
    
    print("=== Starting CBMLoss Framework Test ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Generating Mock Dataloaders...")
    batch_size = 32
    train_loader, val_loader = generate_mock_dataloaders(batch_size=batch_size)
    
    # 2. Initialize Model
    num_concepts = 3
    num_classes = 2 # Healthy or Diseased
    
    print("Initializing Model...")
    model = ConceptBottleneckModel(num_concepts=num_concepts, num_classes=num_classes, backbone_name='resnet18', pretrained=False)
    model = model.to(device)
    
    # 3. Setup Loss and Optimizer
    print("Setting up Loss and Optimizer...")
    # Using lambda weights as specified in standard settings (can be tuned)
    criterion = CBMLoss(lambda_concept=1.0, lambda_ent=0.1, lambda_ortho=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 4. Train Model
    print("Training Model...")
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device)
    trainer.fit(train_loader, val_loader, epochs=3, project_name="cbm-mock-test")
    
    # 5. Evaluate Concept Intervention
    print("Evaluating Causal Concept Intervention...")
    df = evaluate_concept_intervention(model, val_loader, device=device)
    print("\nIntervention Results:")
    print(df)
    
    print("=== Pipeline Test Complete ===")

if __name__ == "__main__":
    main()
