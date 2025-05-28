# Estendendo o VAEP para Avaliação de Cruzamentos no Futebol

Este projeto tem como objetivo estender o framework **VAEP (Valuing Actions by Estimating Probabilities)** para melhorar a avaliação de cruzamentos no futebol. A proposta foca em capturar melhor o valor ofensivo e o risco associado a cruzamentos, considerando seu contexto espacial e tático.

## Objetivo

Adaptar e expandir o modelo VAEP para:

- Identificar cruzamentos e ações associadas (passes para trás, movimentações, finalizações).
- Avaliar o impacto dessas ações na probabilidade de marcar ou sofrer gols.
- Fornecer uma métrica mais precisa e contextualizada para a qualidade de cruzamentos.

## Metodologia

- Uso de dados de eventos (e.g., StatsBomb) para identificar cruzamentos e jogadas relacionadas.
- Treinamento de modelos de machine learning para estimar transições de estado (probabilidades de gol e sofrer gol).
- Cálculo do valor de ações com base na diferença de valor esperado entre estados.
- Extensão das features para capturar aspectos espaciais e sequenciais dos cruzamentos.

## Estrutura do Projeto

```
vaep-cruzamentos/
├── data/                # Scripts e amostras de dados de eventos
├── src/                 # Código-fonte principal (pré-processamento, modelo VAEP estendido)
├── notebooks/           # Análises exploratórias e testes de modelos
├── results/             # Saídas dos experimentos e gráficos
├── requirements.txt     # Dependências do projeto
└── README.md            # Este arquivo
```

## Requisitos

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- xgboost (ou outro modelo de classificação)
- [socceraction](https://github.com/ML-KULeuven/socceraction)

Instale com:

```bash
pip install -r requirements.txt
```

## Como Rodar

1. Prepare os dados de eventos (e.g., StatsBomb).
2. Execute os scripts de pré-processamento em `src/preprocessing/`.
3. Treine os modelos VAEP com as novas features em `src/model/`.
4. Gere os valores de ação com `src/evaluation/`.

## Referências

- Decroos, T. et al. (2019). [Actions Speak Louder than Goals: Valuing Player Actions in Soccer](https://arxiv.org/abs/1802.07127)
- StatsBomb Open Data

## Licença

MIT
