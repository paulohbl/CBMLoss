import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

output_dir = "figs"
data_dir = "../trainamentoOutput"

# Filenames extracted from the directory
files = {
    0.1: "intervention_cub200_resnet18_ent0.1_ortho0.1_seed42_cub200_resnet18_ent0.1_ortho0.1_seed42_best_acc0.7235.csv",
    0.3: "intervention_cub200_resnet18_ent0.3_ortho0.3_seed42_cub200_resnet18_ent0.3_ortho0.3_seed42_best_acc0.6793.csv",
    0.5: "intervention_cub200_resnet18_ent0.5_ortho0.5_seed42_cub200_resnet18_ent0.5_ortho0.5_seed42_best_acc0.6431.csv",
    0.7: "intervention_cub200_resnet18_ent0.7_ortho0.7_seed42_cub200_resnet18_ent0.7_ortho0.7_seed42_best_acc0.6001.csv"
}

data = {}
for lam, fname in files.items():
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        data[lam] = pd.read_csv(path)

# Ensure figures look academic
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.autolayout': True
})

# Plot 1: Random vs Uncertainty for Lambda 0.5 (The Paradox)
if 0.5 in data:
    df_05 = data[0.5]
    plt.figure(figsize=(8, 6))
    plt.plot(df_05["Rate"] * 100, df_05["Random_Acc"], marker='o', linestyle='-', linewidth=3, label="Random (Control)", color="#1f77b4")
    plt.plot(df_05["Rate"] * 100, df_05["Uncertainty_Acc"], marker='s', linestyle='--', linewidth=3, label="Uncertainty (Targeted)", color="#d62728")
    
    plt.fill_between(df_05["Rate"] * 100, df_05["Random_Acc"], df_05["Uncertainty_Acc"], color='gray', alpha=0.15)
    
    plt.title(r"Causal Intervention: The Leakage/Uncertainty Paradox ($\lambda=0.5$)")
    plt.xlabel("Percentage of Intervened Concepts (%)")
    plt.ylabel("Task Accuracy")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "fig_uncertainty_paradox.eps"), format='eps', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig_uncertainty_paradox.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

# Plot 2: Intervention Resilience Across Lambdas (Random Only)
plt.figure(figsize=(9, 6))
colors = {0.1: "#1f77b4", 0.3: "#ff7f0e", 0.5: "#2ca02c", 0.7: "#d62728"}
markers = {0.1: 'o', 0.3: 'v', 0.5: 's', 0.7: 'D'}

for lam in sorted(data.keys()):
    df = data[lam]
    plt.plot(df["Rate"] * 100, df["Random_Acc"], marker=markers[lam], linestyle='-', linewidth=2.5, label=rf"$\lambda={lam}$", color=colors[lam])

plt.title("Constraint Strength ($\lambda$) vs. Intervention Resilience")
plt.xlabel("Random Concept Intervention (%)")
plt.ylabel("Task Accuracy")
plt.legend(title="Regularization")
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig(os.path.join(output_dir, "fig_lambda_ablation.eps"), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "fig_lambda_ablation.pdf"), format='pdf', bbox_inches='tight')
plt.close()

print(f"Generated figures in 'artigo/{output_dir}'")
