📘 Inteligências Artificiais Generativas Baseadas em LLMs
Implementação prática com Mistral-7B e QLoRA

Este repositório contém os códigos desenvolvidos no Trabalho de Conclusão de Curso intitulado “Inteligências Artificiais Generativas Baseadas em LLMs: Estudo de Caso e Perspectivas”, realizado na Universidade Federal de Ouro Preto (UFOP).

O projeto demonstra, na prática, como realizar o fine-tuning eficiente de um Large Language Model (LLM) utilizando a técnica QLoRA, com foco na especialização do modelo em literatura científica sobre LLMs.

🎯 Objetivo do Projeto

O objetivo principal é demonstrar que é possível realizar o treinamento especializado de um modelo de linguagem de larga escala utilizando recursos computacionais limitados (GPU RTX A4000 – 16GB VRAM), ajustando apenas uma pequena fração dos parâmetros do modelo base.

O modelo utilizado foi o:

Mistral-7B-v0.3

Ajustado com LoRA (Low-Rank Adaptation)

Quantização em 4 bits (QLoRA)

🖥️ Requisitos
Hardware recomendado

GPU NVIDIA com mínimo de 16GB de VRAM

CUDA 12.x

32GB de RAM recomendados

O treinamento foi realizado em:

NVIDIA RTX A4000 (16GB VRAM)

📦 Bibliotecas necessárias

Instale as dependências com:

pip install torch transformers peft bitsandbytes datasets accelerate


Caso utilize Windows com CUDA, verifique se a versão do PyTorch está compatível com sua instalação CUDA.

📂 Estrutura do Projeto
/data
    ├── artigos_pdf/
    ├── dataset.jsonl
/scripts
    ├── extract_pdf.py
    ├── preprocess.py
    ├── chunk_text.py
    ├── train.py
    ├── inference.py
/models
    ├── checkpoints/
README.md

⚙️ Passo a Passo de Execução
1️⃣ Extração de Texto dos PDFs

Os artigos científicos devem ser colocados na pasta:

/data/artigos_pdf/


Execute:

python scripts/extract_pdf.py


Esse script:

Lê os PDFs

Remove criptografia quando possível

Extrai o texto bruto

2️⃣ Pré-processamento

Execute:

python scripts/preprocess.py


Esse script realiza:

Remoção de caracteres inválidos

Correção de hifens de quebra de linha

Normalização de espaços

Segmentação em blocos (chunks)

3️⃣ Geração do Dataset (JSONL)

O dataset é estruturado no formato:

[INST] instrução [/INST] resposta


O arquivo final gerado será:

/data/dataset.jsonl

4️⃣ Treinamento do Modelo

Para iniciar o fine-tuning:

python scripts/train.py

Configurações utilizadas no TCC:

Batch size: 4

Gradient accumulation: 2

Learning rate: 1.5e-5

Max sequence length: 2048

Steps: 1000

LoRA rank (r): 16

Alpha: 32

Dropout: 0.05

Quantização: 4-bit (nf4)

Apenas 0,09% dos parâmetros do modelo foram ajustados.

Tempo médio de treinamento:
≈ 7 horas

5️⃣ Inferência

Após o treinamento, execute:

python scripts/inference.py


Formato esperado de entrada:

[INST] Sua pergunta aqui [/INST]


Parâmetros de geração recomendados:

temperature = 0.7

top_p = 0.9

repetition_penalty = 1.1

📊 Avaliação

A avaliação realizada no TCC foi qualitativa, comparando:

Modelo base (sem fine-tuning)

Modelo especializado (com LoRA)

Observou-se melhoria significativa em:

Coerência técnica

Uso de terminologia acadêmica

Referenciação conceitual

Limitações:

Dataset relativamente pequeno (873 exemplos)

Dependência da qualidade das instruções

Treinamento restrito por hardware

⚠️ Observações Importantes

A qualidade do dataset influencia diretamente o comportamento do modelo.

Exemplos mal estruturados podem induzir respostas indesejadas.

Este projeto tem finalidade acadêmica e experimental.

📚 Referência Acadêmica

Caso utilize este código, cite o trabalho:

Bosada Júnior, Ângelo César.
Inteligências Artificiais Generativas Baseadas em LLMs: Estudo de Caso e Perspectivas.
Universidade Federal de Ouro Preto, 2026.
