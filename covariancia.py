import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Fixamos a semente para os resultados serem replicáveis no seu debug
torch.manual_seed(42)

# ==========================================
# 1. DATASET SINTÉTICO (Toy Dataset)
# ==========================================
class ToyDataset(Dataset):
    def __init__(self, num_samples=4):
        """
        Cria 4 amostras. 
        - x: Um vetor aleatório de tamanho 5 (simulando as features de uma imagem).
        - c_true: 3 conceitos visuais binários (ex: 1=Tem halo, 0=Não tem).
        - y_true: A classe final (0 ou 1). Criamos uma regra lógica: se a soma
                  dos conceitos for >= 2, a classe é 1 (Doente). Senão, é 0 (Saudável).
        """
        self.x = torch.randn(num_samples, 5)
        self.c_true = torch.randint(0, 2, (num_samples, 3)).float()
        self.y_true = (self.c_true.sum(dim=1) >= 2).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.c_true[idx], self.y_true[idx]

# ==========================================
# 2. ARQUITETURA DO MODELO (Toy CBM)
# ==========================================
class ToyCBM(nn.Module):
    def __init__(self):
        super().__init__()
        # Extrai 3 conceitos a partir das 5 features da imagem
        self.concept_extractor = nn.Sequential(
            nn.Linear(5, 3),
            nn.Sigmoid() # Força a saída para o intervalo [0, 1]
        )
        # Prevê 2 classes finais (Saudável/Doente) a partir dos 3 conceitos
        self.label_predictor = nn.Linear(3, 2)

    def forward(self, x):
        c_hat = self.concept_extractor(x)
        logits = self.label_predictor(c_hat)
        return logits, c_hat

# ==========================================
# 3. FUNÇÃO DE AVALIAÇÃO COM DEBUG VISUAL
# ==========================================
@torch.no_grad()
def evaluate_concept_intervention_debug(model, dataloader):
    model.eval()
    
    # Pegamos apenas o primeiro batch para debugar
    x, c_true, y_true = next(iter(dataloader))
    
    print("--- INÍCIO DO DEBUG: DADOS REAIS ---")
    print(f"Features da Imagem (x): shape {x.shape}")
    print(f"Conceitos Reais (c_true):\n{c_true}")
    print(f"Classe Real (y_true): {y_true}\n")

    # Passo 1: O Modelo age sozinho
    print("--- PASSO 1: INFERÊNCIA PADRÃO (Sem Intervenção) ---")
    c_hat = model.concept_extractor(x)
    logits_standard = model.label_predictor(c_hat)
    preds_standard = torch.argmax(logits_standard, dim=1)
    
    print("Conceitos Previstos pela IA (c_hat):")
    print(c_hat) # Valores quebrados (floats) onde o leakage acontece
    print(f"Previsão Final da IA: {preds_standard}")
    print(f"Acertos: {(preds_standard == y_true).sum().item()} de {len(y_true)}\n")

    # Passo 2: O Especialista intervem
    print("--- PASSO 2: INTERVENÇÃO CAUSAL (Injetando Ground Truth) ---")
    # Aqui substituímos o c_hat (imperfeito) pelo c_true (perfeito)
    logits_intervened = model.label_predictor(c_true)
    preds_intervened = torch.argmax(logits_intervened, dim=1)
    
    print("Conceitos Injetados (c_true):")
    print(c_true) # Valores estritamente 0 ou 1
    print(f"Nova Previsão Final: {preds_intervened}")
    print(f"Acertos após intervenção: {(preds_intervened == y_true).sum().item()} de {len(y_true)}\n")

    print("--- CONCLUSÃO DO TESTE ---")
    if (preds_intervened == y_true).sum().item() > (preds_standard == y_true).sum().item():
        print("A intervenção MELHOROU o resultado! (O modelo respeita os conceitos).")
    else:
        print("A intervenção NÃO AJUDOU ou PIOROU. (Sintoma clássico de Concept Leakage ou modelo não treinado).")

# ==========================================
# 4. EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    # Inicializa dataset com 4 amostras e um dataloader que pega todas de uma vez
    dataset = ToyDataset(num_samples=4)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Inicializa o modelo não treinado
    model = ToyCBM()
    
    # Roda o debug
    evaluate_concept_intervention_debug(model, dataloader)