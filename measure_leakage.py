import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataset import get_dataloaders
from model import ConceptBottleneckModel

def parse_args():
    parser = argparse.ArgumentParser(description="Direct Concept Leakage Measurement via Linear Probing and Discretization Gap")
    parser.add_argument("--dataset", type=str, default="cub200", choices=["mock", "synthetic_leaf", "cub200"])
    parser.add_argument("--checkpoint_dir", type=str, default="G:/Meu Drive/CBMLoss_Checkpoints",
                        help="Directory where checkpoints (.pth) are stored.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", type=str, default="leakage_direct_metrics.csv")
    return parser.parse_args()

def extract_features(model, dataloader, device):
    """
    Extracts continuous concepts (c_hat), discrete concepts (c_pred_hard),
    ground truth concepts (c_true), residual error (r = c_hat - c_true),
    soft task predictions (y_hat_soft), and discrete task predictions (y_hat_hard).
    """
    model.eval()
    all_c_hat = []
    all_c_true = []
    all_labels = []
    all_y_soft = []
    all_y_hard = []

    with torch.no_grad():
        for images, concepts, labels in tqdm(dataloader, desc="Extracting concept activations", leave=False):
            images = images.to(device)
            concepts = concepts.to(device)
            
            c_hat = model.extractor(images)
            y_soft = model.predictor(c_hat)
            
            # Predict labels from binarized concept predictions
            c_hard = (c_hat >= 0.5).float()
            y_hard = model.predictor(c_hard)
            
            all_c_hat.append(c_hat.cpu())
            all_c_true.append(concepts.cpu())
            all_labels.append(labels)
            all_y_soft.append(torch.argmax(y_soft, dim=1).cpu())
            all_y_hard.append(torch.argmax(y_hard, dim=1).cpu())

    c_hat_all = torch.cat(all_c_hat, dim=0).numpy()
    c_true_all = torch.cat(all_c_true, dim=0).numpy()
    labels_all = torch.cat(all_labels, dim=0).numpy()
    y_soft_all = torch.cat(all_y_soft, dim=0).numpy()
    y_hard_all = torch.cat(all_y_hard, dim=0).numpy()

    return c_hat_all, c_true_all, labels_all, y_soft_all, y_hard_all

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"=== Direct Leakage Evaluation on {args.dataset.upper()} using device: {device} ===")

    # 1. Load Data
    _, val_loader, num_concepts, num_classes = get_dataloaders(args.dataset, batch_size=args.batch_size)

    # 2. Identify Checkpoint Files
    if not os.path.exists(args.checkpoint_dir):
        # Fallback to local checkpoints folder if path not found
        args.checkpoint_dir = "checkpoints"
        
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint dir '{args.checkpoint_dir}' does not exist.")
        return

    checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith("_best.pth")]
    print(f"Found {len(checkpoint_files)} checkpoints in {args.checkpoint_dir}")

    results = []

    for fname in sorted(checkpoint_files):
        fpath = os.path.join(args.checkpoint_dir, fname)
        print(f"\nEvaluating: {fname}")

        # Instantiate Model
        model = ConceptBottleneckModel(num_concepts=num_concepts, num_classes=num_classes, backbone_name="resnet18", pretrained=False)
        checkpoint = torch.load(fpath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # Extract features
        c_hat, c_true, labels, y_soft, y_hard = extract_features(model, val_loader, device)

        # 1. Soft Accuracy vs Discrete Accuracy (Discretization Gap)
        acc_soft = accuracy_score(labels, y_soft)
        acc_hard = accuracy_score(labels, y_hard)
        leakage_gap = acc_soft - acc_hard  # Higher gap = model relied on continuous shortcut!

        # 2. Linear Probe on Residual Concept Noise: r = c_hat - c_true
        # In a pure model, r contains no class information. In a leaky model, r encodes class cues.
        residuals = c_hat - c_true
        
        # Split train/test for probe (70/30 split of val set)
        n = len(labels)
        perm = np.random.RandomState(42).permutation(n)
        split = int(0.7 * n)
        train_idx, test_idx = perm[:split], perm[split:]

        probe = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        probe.fit(residuals[train_idx], labels[train_idx])
        probe_acc = probe.score(residuals[test_idx], labels[test_idx])

        # Extract lambda values from filename if present
        ent_val = None
        ortho_val = None
        for part in fname.split("_"):
            if part.startswith("ent"):
                try: ent_val = float(part.replace("ent", ""))
                except: pass
            elif part.startswith("ortho"):
                try: ortho_val = float(part.replace("ortho", ""))
                except: pass

        results.append({
            "Checkpoint": fname,
            "lambda_ent": ent_val,
            "lambda_ortho": ortho_val,
            "Soft_Task_Acc": acc_soft,
            "Binarized_Task_Acc": acc_hard,
            "Continuous_Leakage_Gap": leakage_gap,
            "Residual_Probe_Acc": probe_acc
        })

        print(f"-> Soft Acc: {acc_soft*100:.2f}% | Binarized Acc: {acc_hard*100:.2f}% | Leakage Gap: {leakage_gap*100:.2f}% | Residual Probe Acc: {probe_acc*100:.2f}%")

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output_csv, index=False)
    print(f"\nSaved direct leakage evaluation results to {args.output_csv}")
    print("\nSummary Table:")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
