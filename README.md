# Projeto de Reconhecimento de Comandos de Voz

Este projeto utiliza algoritmos de aprendizado de máquina para reconhecer comandos de voz a partir de áudios curtos. Abaixo está uma explicação geral da estrutura dos arquivos e pastas do repositório.

## Estrutura do projeto

- **Algorithms/**
  - Contém implementações dos principais algoritmos utilizados:
    - `Neural Network/`: Algoritmos baseados em redes neurais.
    - `KNN/`: Algoritmos baseados em K-Nearest Neighbors.
    - `SVM/`: Algoritmos baseados em Support Vector Machines.
    - `Artigo 1/`: Algoritmos e experimentos relacionados ao nosso artigo escolhido.
  - Cada subpasta geralmente possui arquivos `__init__.py` (implementação principal) e `test.py` (testes). A pasta `Artigo 1` também possui um arquivo `robo.py`, que simula um robô ouvindo em tempo real of comandos de voz, pode ser testado pela professora!

- **files/**
  - **dataset/**: Deve conter o conjunto de dados de comandos de voz (Speech Commands Dataset v0.02), organizado em subpastas por palavra (ex: `forward/`, `right/`, `left/`, etc).
  - **models/**: Modelos treinados e arquivos de mapeamento de rótulos para cada algoritmo:
    - Subpastas para cada abordagem (`Neural Network/`, `SVM/`, `KNN/`, `Artigo 1/`), contendo arquivos como `voice_model.pt`, `svm_model.joblib`, `knn_model.joblib` e arquivos de mapeamento (`label_mapping.json`) dos labels.
    - Arquivos na raiz da pasta para modelos específicos (ex: `voice_model_yes_no_only.pt`).
  - **recorded/**: Áudios gravados por nós (Ari e Luigi) para testes, organizados por nome (ex: `ari/`, `luigi/`). Cada pasta contém arquivos `.wav` com comandos gravados.


## Resumo

O projeto está organizado para facilitar o desenvolvimento, teste e comparação de diferentes algoritmos de reconhecimento de comandos de voz. Os dados estão separados dos modelos e dos códigos, e há exemplos de áudios gravados por nós para validação prática.

Para ler sobre o dataset, `files/dataset/README.md`. 