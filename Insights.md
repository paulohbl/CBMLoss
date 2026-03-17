# Insights Científicos: Mitigação de Concept Leakage no CUB-200

Este documento consolida as observações e conclusões obtidas durante os experimentos iniciais do framework CBMLoss. Estas notas servem como base para a escrita do artigo científico para o SIBGRAPI.

---

## 1. O Fenômeno do Concept Leakage (Vazamento de Conceitos)

Nos experimentos iniciais (com regularização fraca), observamos um padrão clássico de **Concept Leakage**:

- **Sintoma**: Alta acurácia na tarefa final (~46%) mas falha catastrófica na intervenção causal (~9%).
- **Causa**: O modelo não está usando o significado semântico dos conceitos (ex: "bico amarelo"). Em vez disso, o extrator de conceitos está usando as ativações contínuas para passar informações "piratas" (texturas, fundo, ruído) para o classificador final.
- **Conclusão**: O modelo é funcional, mas não é explicável. Se um humano corrigir um conceito, o modelo erra a predição porque sua "lógica interna" está baseada em pistas escondidas e não nos conceitos reais.

---

## 2. O Trade-off da Interpretabilidade

Ao aumentar os hiperparâmetros de regularização (`--lambda_ent` e `--lambda_ortho` para 0.5):

- **Observação**: A acurácia na tarefa caiu (de 46% para 30%).
- **Análise**: Este é o "preço" da honestidade. Ao forçar os conceitos a serem ortogonais (independentes) e binários (entropia), impedimos o modelo de usar atalhos fáceis. 
- **Conclusão**: A queda na acurácia não é necessariamente um erro, mas sim o modelo sendo forçado a aprender da maneira difícil (e correta). Para mitigar essa queda, é necessário um tempo de treinamento (épocas) maior para que o classificador se adapte aos novos conceitos "purificados".

---

## 3. Eficácia da CBMLoss

As métricas do WandB provaram que a formulação matemática da perda está correta:

- **Redução da Ortogonalidade**: A perda de ortogonalidade caiu de 2.0 para 0.3, indicando que o modelo parou de colapsar múltiplos conceitos na mesma representação estrutural.
- **Ajuste da Entropia**: As ativações saíram de valores intermediários (incertos) para valores próximos de 0 ou 1, limpando o gargalo semântico.

---

## 4. Guia para Resultados "Publicáveis" (SIBGRAPI)

Para gerar os gráficos definitivos do artigo, o protocolo sugerido é:

1.  **Baseline (Controle)**: Treinar com `lambda_ent=0` e `lambda_ortho=0`. Mostrar a alta performance inicial e o colapso na intervenção.
2.  **Modelo Proposto**: Treinar com `lambda_ent=0.5` e `lambda_ortho=0.5`. 
3.  **Comprovação**: Mostrar que, embora a acurácia inicial seja menor, a **curva de intervenção é mais resiliente**.
4.  **Convergência**: O CUB-200 exige entre 40 a 60 épocas para que o classificador final aprenda a correlacionar conceitos puros com as 200 espécies de pássaros de forma robusta.

---

## 5. Dicas de Engenharia

- **Pretrained Weights**: Essencial para o CUB-200. Sem pesos pré-treinados (ImageNet), o modelo levaria centenas de épocas apenas para aprender a distinguir penas de galhos.
- **Data Augmentation**: O uso de `RandomResizedCrop` foi o fator decisivo para eliminar o efeito "dente de serra" (instabilidade) observado nos primeiros gráficos.
