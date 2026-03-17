# CBMLoss: Mitigating Concept Leakage in Concept Bottleneck Models

Este framework foi desenvolvido para a pesquisa de mitigação de *Concept Leakage* em *Concept Bottleneck Models* (CBMs), focando no uso de perdas de Entropia e Ortogonalidade para garantir a pureza semântica dos conceitos.

## 🚀 Estrutura do Projeto

- `main.py`: Ponto de entrada para treinamento e avaliação.
- `model.py`: Definições da arquitetura CBM (Concept Extractor + Label Predictor).
- `losses.py`: Implementação da **CBMLoss** (Fidelity + Entropy + Orthogonality).
- `dataset.py`: Dataloaders para CUB-200, Synthetic Leaf e Mock data.
- `train.py`: Loop de treinamento, validação e checkpointing.
- `evaluate.py`: Script de intervenção causal para medir leakage.
- `download_datasets.py`: Script para preparar os dados.

---

## 🛠️ Instalação

### Usando Conda (Recomendado)
```bash
conda create -n CBMLoss python=3.10
conda activate CBMLoss
pip install -r requirements.txt
```

### Usando Pip
```bash
pip install torch torchvision torchaudio wandb matplotlib pandas scikit-learn tqdm
```

---

## 📊 Preparação dos Dados

Antes de treinar, você precisa baixar ou gerar os datasets:
```bash
python download_datasets.py
```
Isso criará a pasta `data/` com o dataset sintético e o CUB-200 formatado.

---

## 🏋️ Treinamento

### Execução Padrão (CUB-200)
```bash
python main.py --dataset cub200 --epochs 60 --lr 1e-4
```

### Argumentos Principais:
- `--dataset`: `cub200`, `synthetic_leaf` ou `mock`.
- `--lambda_ent`: Peso da regularização de Entropia (Mitigação de Leakage).
- `--lambda_ortho`: Peso da regularização de Ortogonalidade.
- `--pretrained`: Usa ResNet18 pré-treinada (Padrão: True).
- `--batch_size`: Tamanho do lote (Padrão: 32).

---

## 💾 Persistência e Resumo (Google Colab)

Para não perder o progresso no Colab, monte o Google Drive e use a flag de checkpoint:

1. **Montar Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Treinar salvando no Drive:**
```bash
!python CBMLoss/main.py --dataset cub200 --epochs 60 --checkpoint_dir "/content/drive/My Drive/CBM_Checkpoints"
```

3. **Retomar treino interrompido:**
```bash
!python CBMLoss/main.py --dataset cub200 --epochs 60 \
  --resume_from "/content/drive/My Drive/CBM_Checkpoints/checkpoint_latest.pth" \
  --wandb_id "ID_DO_WANDB_AQUI"
```

---

## 🧪 Avaliação e Intervenção

Ao final de cada treino, o script `evaluate.py` é chamado automaticamente para realizar a **Intervenção Causal**. Ele substitui os conceitos previstos pelo modelo pelos conceitos Reais (Ground Truth) em diferentes níveis (0% a 100%) e gera o gráfico `intervention_results.png`.

Se a acurácia subir drasticamente com a intervenção, o modelo é um CBM robusto. Se a acurácia mudar pouco, pode haver *Concept Leakage* (o modelo está ignorando os conceitos e usando pistas ocultas).

---

## 📓 Pesquisa e Insights Científicos

Para uma análise detalhada sobre o fenômeno de **Concept Leakage**, o **Trade-off de Interpretabilidade** e como interpretar os gráficos de intervenção gerados para o artigo do SIBGRAPI, consulte o arquivo:

---

## 📊 Monitoramento (WandB)

O framework é integrado ao Weights & Biases. Você pode acompanhar as perdas de cada componente (`loss/leakage_entropy`, `loss/leakage_ortho`, etc.) em tempo real pelo dashboard.
