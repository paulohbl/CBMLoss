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
    
    Generates two curves for comparison: 
    1. Random Intervention
    2. Uncertainty-based Intervention (least confident first)
    """
    if intervention_rates is None:
        intervention_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
    model.eval()
    model.to(device)
    
    modes = ["Random", "Uncertainty"]
    all_results = {mode: [] for mode in modes}
    
    with torch.no_grad():
        for mode in modes:
            print(f"Running {mode} Intervention Evaluation...")
            for rate in intervention_rates:
                task_correct = 0
                total_samples = 0
                
                for images, concepts_gt, labels in tqdm(dataloader, desc=f"{mode} {rate:.0%}", leave=False):
                    images = images.to(device)
                    concepts_gt = concepts_gt.to(device)
                    labels = labels.to(device)
                    
                    # 1. Forward pass extracting concepts only
                    c_hat = model.extractor(images)
                    batch_size, num_concepts = c_hat.shape
                    
                    # 2. Intervene: Create a mask of which concepts to replace
                    if mode == "Random":
                        # We intervene on a random subset of concepts per sample
                        mask = torch.rand(batch_size, num_concepts, device=device) < rate
                    else: # Uncertainty
                        # Calculate uncertainty: how close c_hat is to 0.5
                        # 0.0 means perfectly 0.5 (max uncertainty), 0.5 means 0 or 1 (max certainty)
                        uncertainty = torch.abs(c_hat - 0.5)
                        
                        # Number of concepts to intervene on this batch
                        k = int(rate * num_concepts)
                        if k == 0:
                            mask = torch.zeros_like(c_hat, dtype=torch.bool)
                        elif k >= num_concepts:
                            mask = torch.ones_like(c_hat, dtype=torch.bool)
                        else:
                            # Select top-k most uncertain (smallest distance to 0.5)
                            _, indices = torch.topk(uncertainty, k=k, dim=1, largest=False)
                            mask = torch.zeros_like(c_hat, dtype=torch.bool)
                            mask.scatter_(1, indices, True)
                    
                    # c_intervened is a mix of ground truth and predicted concepts
                    c_intervened = torch.where(mask, concepts_gt, c_hat)
                    
                    # 3. Pass intervened concepts to label predictor
                    y_hat = model.predictor(c_intervened)
                    
                    preds = torch.argmax(y_hat, dim=1)
                    task_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    
                acc = task_correct / total_samples
                all_results[mode].append(acc)
                
    # Plotting Comparison
    plt.figure(figsize=(10, 7))
    
    for mode in modes:
        marker = 'o' if mode == "Random" else 's'
        color = 'blue' if mode == "Random" else 'red'
        plt.plot(np.array(intervention_rates) * 100, all_results[mode], 
                 marker=marker, color=color, linestyle='-', linewidth=2.5, label=f"{mode} Intervention")
        
    plt.title("Concept Intervention Comparison: Random vs. Uncertainty", fontsize=14, fontweight='bold')
    plt.xlabel("Percentage of Intervened Concepts (%)", fontsize=12)
    plt.ylabel("Task Accuracy", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    
    # Text annotation for Causal Gap
    max_acc = max(all_results["Random"][0], all_results["Uncertainty"][0])
    min_acc_random = all_results["Random"][-1]
    plt.annotate(f'Causal Gap: {max_acc-min_acc_random:.2f}', 
                 xy=(100, min_acc_random), xytext=(70, min_acc_random + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison Plot saved as '{plot_path}'")
    
    # Return a dataframe for summary
    return pd.DataFrame({
        "Rate": intervention_rates,
        "Random_Acc": all_results["Random"],
        "Uncertainty_Acc": all_results["Uncertainty"]
    })
