import torch
import transformers
import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

# Configurar para usar a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Força o uso da GPU 0

# 1. Configurações
MODEL_PATH = "./mistral-7b-v0.3"
DATA_PATH = "./training_data.jsonl"
OUTPUT_DIR = "./results"

# 2. Verificar GPU
print(f"GPU disponível: {torch.cuda.is_available()}")
print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
print(f"Memória VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# 3. Configuração de Quantização 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 4. Carregar Modelo com Quantização
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# 5. Preparar modelo para treino 4-bit
model = prepare_model_for_kbit_training(model)

# 6. Configuração LoRA (Otimizada para 6GB)
peft_config = LoraConfig(
    r=16,                  # Rank mais baixo para economizar memória
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Apenas módulos mais importantes
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 7. Aplicar LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 8. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# 9. Carregar Dataset
dataset = load_dataset('json', data_files=DATA_PATH, split='train')

# 10. Tokenização com truncamento
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,  # Contexto menor para economizar memória
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 11. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 12. Configuração de Treinamento (Otimizada para 6GB)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,      # Aumenta para 4 ou 6 — usa mais VRAM, treino mais rápido
    gradient_accumulation_steps=2,      # Menos acumulação — aproveita batch maior
    learning_rate=1.5e-5,                 # Taxa de aprendizado mais segura para batch maior
    weight_decay=0.01,
    logging_steps=5,
    max_steps=1000,
    fp16=True,                          # Mantém fp16 (A4000 lida bem)
    optim="paged_adamw_8bit",
    save_strategy="no",
    report_to="none",
    ddp_find_unused_parameters=False,
    remove_unused_columns=True
)


# 13. Criar Trainer
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 14. Treinar
print("Iniciando treinamento na GPU...")
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
    trainer.train()

# 15. Salvar adaptadores LoRA
model.save_pretrained(OUTPUT_DIR)
print(f"Treinamento concluído! Adaptadores salvos em {OUTPUT_DIR}")