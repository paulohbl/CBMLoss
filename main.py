import torch
import torch.optim as optim
import argparse
import wandb
import os

from dataset import get_dataloaders
from model import ConceptBottleneckModel
from losses import CBMLoss
from train import Trainer
from evaluate import evaluate_concept_intervention

def parse_args():
    parser = argparse.ArgumentParser(description="Train CBM with Leakage Mitigation")
    
    # Dataset and Framework setup
    parser.add_argument("--dataset", type=str, choices=["mock", "synthetic_leaf", "cub200"], default="mock",
                        help="Choose which dataset to run (run download_datasets.py first for real data).")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Pretrained backbone")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pre-trained ResNet weights from ImageNet (Highly Recommended)")
    
    # Loss Regularization Hyperparameters (The Contribution)
    parser.add_argument("--lambda_concept", type=float, default=1.0, help="Weight for Concept BCELoss")
    parser.add_argument("--lambda_ent", type=float, default=0.1, help="Weight for Concept Entropy Loss (Leakage mitigation)")
    parser.add_argument("--lambda_ortho", type=float, default=0.1, help="Weight for Concept Orthogonality Loss (Leakage mitigation)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging and Persistance
    parser.add_argument("--offline", action="store_true", help="Run WandB offline")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a .pth checkpoint to resume from")
    parser.add_argument("--wandb_id", type=str, default=None, help="WandB run ID to resume (e.g. f7aun4ej)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        
    print(f"=== Starting CBMLoss Framework Test on {args.dataset.upper()} Dataset ===")
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 0. Environment Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading dataloaders for {args.dataset}...")
    try:
        train_loader, val_loader, num_concepts, num_classes = get_dataloaders(args.dataset, batch_size=args.batch_size)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please run: conda run -n CBMLoss python download_datasets.py")
        return
        
    print(f"Dataset specs | Concepts: {num_concepts} | Classes: {num_classes}")
    
    # 2. Initialize Model
    print("Initializing Model...")
    model = ConceptBottleneckModel(num_concepts=num_concepts, num_classes=num_classes, backbone_name=args.backbone, pretrained=args.pretrained)
    model = model.to(device)
    
    # 3. Setup Loss, Optimizer and Scheduler
    print(f"Setting up Loss -> lambda_ent: {args.lambda_ent}, lambda_ortho: {args.lambda_ortho}")
    criterion = CBMLoss(lambda_concept=args.lambda_concept, lambda_ent=args.lambda_ent, lambda_ortho=args.lambda_ortho)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 4. Handle Resumption
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device)
    start_epoch = 0
    
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from, scheduler=scheduler)
    
    # WandB Config
    wandb_config = vars(args)
    wandb_config["num_concepts"] = num_concepts
    wandb_config["num_classes"] = num_classes
    
    # Initialize/Resume WandB
    run = wandb.init(
        project="cbm-leakage-mitigation", 
        config=wandb_config,
        id=args.wandb_id,
        resume="allow"
    )
    wandb.config.update({"model": model.__class__.__name__}, allow_val_change=True)
    
    # 5. Train Model
    print("Training Model...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate_epoch(val_loader)
        
        scheduler.step()
        
        # Save local checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_latest.pth")
        trainer.save_checkpoint(epoch + 1, ckpt_path, scheduler=scheduler)
        
        # Log to wandb
        wandb.log({
            **train_metrics, 
            **val_metrics, 
            "epoch": epoch+1, 
            "lr": optimizer.param_groups[0]['lr']
        })
        print(f"Train Loss: {train_metrics['loss/total']:.4f} | Task Acc: {val_metrics['val/task_accuracy']:.4f} | Concept Acc: {val_metrics['val/concept_accuracy']:.4f}")
        
    wandb.finish()
    
    # 6. Evaluate Concept Intervention
    print("Evaluating Causal Concept Intervention...")
    df = evaluate_concept_intervention(model, val_loader, device=device)
    print("\nIntervention Results:")
    print(df)
    
    print("=== Pipeline Test Complete ===")

if __name__ == "__main__":
    main()
