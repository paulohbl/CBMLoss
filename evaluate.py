import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def evaluate_concept_intervention(model: torch.nn.Module, dataloader, device: torch.device, intervention_rates: list = None, plot_path: str = "intervention_results.png"):
    """
    Evaluates Concept Leakage by intervening on a percentage of predicted concepts.
    Replaces predicted concepts (c_hat) with ground truth concepts (c) before passing
    them to the LabelPredictor.
    
    If accuracy doesn't increase with intervention, the model suffers from Concept Leakage.
    """
    if intervention_rates is None:
        intervention_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
    model.eval()
    model.to(device)
    
    results = []
    
    with torch.no_grad():
        for rate in intervention_rates:
            task_correct = 0
            total_samples = 0
            
            for images, concepts_gt, labels in tqdm(dataloader, desc=f"Intervening {rate:.0%}", leave=False):
                images = images.to(device)
                concepts_gt = concepts_gt.to(device)
                labels = labels.to(device)
                
                # 1. Forward pass extracting concepts only
                c_hat = model.extractor(images)
                
                # 2. Intervene: Create a mask of which concepts to replace
                batch_size, num_concepts = c_hat.shape
                
                # We intervene on a random subset of concepts per sample
                mask = torch.rand(batch_size, num_concepts, device=device) < rate
                
                # c_intervened is a mix of ground truth and predicted concepts
                c_intervened = torch.where(mask, concepts_gt, c_hat)
                
                # 3. Pass intervened concepts to label predictor
                y_hat = model.predictor(c_intervened)
                
                preds = torch.argmax(y_hat, dim=1)
                task_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
            acc = task_correct / total_samples
            results.append({"Intervention Rate": rate, "Task Accuracy": acc})
            
    df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(df["Intervention Rate"] * 100, df["Task Accuracy"], marker='o', linestyle='-', linewidth=2)
    plt.title("Concept Intervention Performance (Causal Concept Leakage Test)")
    plt.xlabel("Percentage of Intervened Concepts (%)")
    plt.ylabel("Task Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved as '{plot_path}'")
    
    return df
