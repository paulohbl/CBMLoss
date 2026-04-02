import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_dataloaders
from model import ConceptBottleneckModel

def find_qualitative_examples():
    """
    Runs both Lambda=0.1 and Lambda=0.5 models over the CUB200 test set
    to find stark contrasts in decision making. 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths assuming Colab mounted drive structure (adjust if running locally)
    checkpoint_dir = "/content/drive/My Drive/CBMLoss_Checkpoints"
    path_01 = os.path.join(checkpoint_dir, "cub200_resnet18_ent0.1_ortho0.1_seed42_cub200_resnet18_ent0.1_ortho0.1_seed42_best.pth")
    path_05 = os.path.join(checkpoint_dir, "cub200_resnet18_ent0.5_ortho0.5_seed42_cub200_resnet18_ent0.5_ortho0.5_seed42_best.pth")
    
    if not os.path.exists(path_01) or not os.path.exists(path_05):
        print("Warning: Please ensure the checkpoint paths map correctly to your environment.")
        # Fallback to local drive for testing if not on colab
        checkpoint_dir = r"C:\Users\IFMT\Meu Drive\CBMLoss_Checkpoints"
        path_01 = os.path.join(checkpoint_dir, "cub200_resnet18_ent0.1_ortho0.1_seed42_cub200_resnet18_ent0.1_ortho0.1_seed42_best.pth")
        path_05 = os.path.join(checkpoint_dir, "cub200_resnet18_ent0.5_ortho0.5_seed42_cub200_resnet18_ent0.5_ortho0.5_seed42_best.pth")

    print("Loading Dataset...")
    _, val_loader, num_concepts, num_classes = get_dataloaders("cub200", batch_size=1)
    
    print("Loading Model 0.1 (Leaky/Overfit)...")
    model_01 = ConceptBottleneckModel(num_concepts, num_classes, "resnet18").to(device)
    ckpt_01 = torch.load(path_01, map_location=device)
    model_01.load_state_dict(ckpt_01.get('model_state_dict', ckpt_01))
    model_01.eval()
    
    print("Loading Model 0.5 (Clean/Robust)...")
    model_05 = ConceptBottleneckModel(num_concepts, num_classes, "resnet18").to(device)
    ckpt_05 = torch.load(path_05, map_location=device)
    model_05.load_state_dict(ckpt_05.get('model_state_dict', ckpt_05))
    model_05.eval()
    
    print("Mining for interesting examples...")
    found_examples = 0
    
    with torch.no_grad():
        for i, (images, concepts, labels) in enumerate(val_loader):
            images, concepts, labels = images.to(device), concepts.to(device), labels.to(device)
            
            y_pred_01, c_pred_01 = model_01(images)
            y_pred_05, c_pred_05 = model_05(images)
            
            # Get actual predictions
            pred_class_01 = torch.argmax(y_pred_01, dim=1).item()
            pred_class_05 = torch.argmax(y_pred_05, dim=1).item()
            true_class = labels.item()
            
            # Let's find a case where Model 0.1 fails but Model 0.5 succeeds
            # This shows the regularization actually helped generalization in this specific instance
            if pred_class_01 != true_class and pred_class_05 == true_class:
                
                # Check concept predictions (binarized)
                c_pred_bin_01 = (c_pred_01 > 0.5).float()
                c_pred_bin_05 = (c_pred_05 > 0.5).float()
                
                c_acc_01 = (c_pred_bin_01 == concepts).float().mean().item()
                c_acc_05 = (c_pred_bin_05 == concepts).float().mean().item()
                
                # We want a case where 0.5 also has strictly better concepts!
                if c_acc_05 > c_acc_01:
                    print(f"\n[FOUND EXAMPLE {found_examples+1}] Index: {i}")
                    print(f"True Class: {true_class}")
                    print(f"Model 0.1 -> Class: {pred_class_01} (WRONG) | Concept Acc: {c_acc_01:.4f}")
                    print(f"Model 0.5 -> Class: {pred_class_05} (RIGHT) | Concept Acc: {c_acc_05:.4f}")
                    
                    # Plot and save
                    img = images[0].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize ImageNet
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(img)
                    plt.title(f"True Bird: {true_class}\nModel 0.1 (Leaky): Guessed {pred_class_01} (C-Acc {c_acc_01:.2f})\nModel 0.5 (Robust): Guessed {pred_class_05} (C-Acc {c_acc_05:.2f})")
                    plt.axis('off')
                    
                    os.makedirs("artigo/figs", exist_ok=True)
                    save_path = f"artigo/figs/qualitative_example_{found_examples+1}.png"
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    print(f"Saved qualitative image to {save_path}")
                    
                    found_examples += 1
                    
            if found_examples >= 3:
                print("Found enough robust examples! Stopping search.")
                break

if __name__ == "__main__":
    find_qualitative_examples()
