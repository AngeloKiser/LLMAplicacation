import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig

# 1. Configurações
BASE_MODEL_PATH = "./mistral-7b-v0.3"  # Caminho do modelo base original
PEFT_MODEL_PATH = "./results"           # Caminho dos adaptadores LoRA treinados

# 2. Configuração de Quantização para inferência
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 3. Carregar modelo base
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# 4. Carregar adaptadores LoRA treinados
model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)

# 5. Carregar tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# 6. Criar pipeline de geração
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16
)

# 7. Função para formatar prompt
def format_prompt(question):
    return f"[INST] {question} [/INST]"

# 8. Função de geração com controle de memória
def generate_response(question, max_new_tokens=7800):
    prompt = format_prompt(question)
    
    # Configurações de geração otimizadas para 6GB VRAM
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extrair apenas a resposta (remover o prompt)
    full_text = output[0]['generated_text']
    response = full_text.split("[/INST]")[-1].strip()
    return response

# 9. Exemplo de uso
if __name__ == "__main__":
    questions = [
        "Explain in detail the Transformer architecture and how the self-attention mechanism replaced RNNs and CNNs in NLP tasks.",
        "What was the main innovation introduced by GPT-1 compared to previous models, and how did pre-training followed by fine-tuning change NLP development?",
        "What are emergent abilities in large language models, and how were they demonstrated in GPT-3?",
        "Explain how reinforcement learning from human feedback (RLHF) works and why it is important for language model safety and usefulness.",
        "What are the main opportunities and risks associated with foundation models, and why do they represent a paradigm shift in AI?",
        "What are the current challenges in the development and application of large language models according to recent surveys?"
    ]

    with open("respostas_modelo.txt", "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            response = generate_response(question)
            print(f"Answer: {response}")
            print("-" * 80)

            # escreve no arquivo
            f.write(f"Question {i}: {question}\n")
            f.write(f"Answer: {response}\n")
            f.write("-" * 80 + "\n\n")