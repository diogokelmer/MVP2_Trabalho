# MVP - Machine Learning & Analytics

## Previsão de Custos de Seguro de Saúde  

Este repositório contém o trabalho de MVP desenvolvido na disciplina de **Machine Learning** da pós-graduação em Data Science & Analytics (PUC-Rio).  
O objetivo foi aplicar um fluxo completo de aprendizado de máquina supervisionado para prever os custos de seguro de saúde a partir de variáveis demográficas e de estilo de vida.  

---

## Dataset  
- **Fonte:** [Kaggle – Insurance Dataset (mirichoi0218)](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
- **Observações:** 1338 registros e 7 colunas.  
- **Variáveis disponíveis:**
  - `age`: idade do beneficiário  
  - `sex`: sexo do beneficiário  
  - `bmi`: índice de massa corporal  
  - `children`: número de dependentes  
  - `smoker`: indicador se é fumante  
  - `region`: região de residência  
  - `charges`: custo individual do seguro (variável alvo)  

---

## Pré-processamento  
- **Numéricas (`age`, `bmi`, `children`)**  
  - Imputação pela mediana  
  - Padronização com `StandardScaler`  
- **Categóricas (`sex`, `smoker`, `region`)**  
  - Imputação pelo valor mais frequente  
  - Codificação com `OneHotEncoder`  
- Toda a preparação foi encapsulada em um **Pipeline do scikit-learn**, garantindo reprodutibilidade e evitando vazamento de dados.  

---

## Modelagem  
1. **Baseline:** `DummyRegressor (mediana)`  
2. **Modelos testados:**  
   - Linear Regression  
   - Ridge Regression  
   - Random Forest Regressor  
   - Gradient Boosting Regressor  
3. **Validação:**  
   - Divisão treino/teste (80/20)  
   - Validação cruzada com `KFold (5 folds)`  
4. **Otimização de hiperparâmetros:**  
   - `RandomizedSearchCV` aplicado principalmente em Random Forest e Gradient Boosting  
   - Métrica de seleção: **RMSE** (erro quadrático médio)  

---

## Resultados  
- **Baseline:** RMSE ≈ 12.9k, R² negativo  
- **Modelos lineares:** RMSE ≈ 5.8k, R² ≈ 0.78  
- **Random Forest:** RMSE ≈ 4.4k, R² ≈ 0.876  
- **Gradient Boosting (melhor modelo):**  
  - MAE ≈ 2443  
  - RMSE ≈ 4327  
  - R² ≈ 0.879  

Gráficos de comparação de métricas, valores reais vs preditos e distribuição de resíduos foram incluídos para análise.  

---

## Conclusões  
- O problema apresenta forte **não linearidade**, capturada principalmente por modelos de árvores/boosting.  
- O **Gradient Boosting Regressor** foi o melhor modelo, explicando ~88% da variância dos custos de seguro.  
- O modelo tem maior dificuldade em prever **valores extremos (outliers)** de `charges`.  
- Apesar do bom desempenho, há **limitações** ligadas ao tamanho reduzido da base e possível viés de amostragem.  

---

## Próximos Passos  
- Criar novas features (ex.: categorias de IMC, interações entre idade e tabagismo).  
- Testar modelos adicionais (XGBoost, LightGBM, CatBoost).  
- Usar técnicas para lidar melhor com outliers (ex.: transformação logarítmica em `charges`).  
- Avaliar explicabilidade com **Permutation Importance/SHAP**.  

---

## Estrutura do repositório  
