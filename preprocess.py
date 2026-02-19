import json
import re
import os
from transformers import AutoTokenizer

# Configurações
CONFIG = {
    "model_path": "./mistral-7b-v0.3",  # Pasta do modelo baixado
    "input_folder": r"C:\Users\c0678507.sp\Documents\Mistral\data",  # Pasta onde estão os TXT
    "output_jsonl": "./training_data.jsonl",
    "max_seq_length": 4096,  # Ajuste conforme sua GPU
    "prompt_template": "[INST] Analise este texto acadêmico: {text} [/INST]"
}

def verify_files():
    """Verifica a existência da pasta e do modelo"""
    if not os.path.isdir(CONFIG['input_folder']):
        raise FileNotFoundError(f"Pasta não encontrada: {CONFIG['input_folder']}")
    
    if not os.path.exists(CONFIG['model_path']):
        raise FileNotFoundError(f"Pasta do modelo não encontrada: {CONFIG['model_path']}")

def load_tokenizer():
    """Carrega o tokenizer com fallback"""
    try:
        return AutoTokenizer.from_pretrained(
            CONFIG['model_path'],
            local_files_only=True
        )
    except Exception as e:
        print(f"⚠️ Erro ao carregar tokenizer rápido: {e}")
        from transformers import LlamaTokenizer  # Fallback
        return LlamaTokenizer.from_pretrained(
            CONFIG['model_path'],
            local_files_only=True,
            legacy=False
        )

def clean_text(text):
    """Limpeza avançada do texto"""
    text = re.sub(r'\s+', ' ', text)  # Espaços duplicados
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Caracteres não imprimíveis
    text = re.sub(r'-\s*\n\s*', '', text)  # Hifens no final da linha
    return text.strip()

def chunk_text(text, tokenizer):
    """Divide o texto em chunks sem quebrar sentenças"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    current_length = 0

    for para in paragraphs:
        tokens = tokenizer.encode(para, add_special_tokens=False)
        para_length = len(tokens)
        
        if para_length > CONFIG['max_seq_length']:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
                sent_length = len(sent_tokens)
                
                if current_length + sent_length <= CONFIG['max_seq_length']:
                    current_chunk += sent + " "
                    current_length += sent_length
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
                    current_length = sent_length
        else:
            if current_length + para_length <= CONFIG['max_seq_length']:
                current_chunk += para + "\n\n"
                current_length += para_length
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                current_length = para_length
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def main():
    try:
        print("🔍 Verificando arquivos...")
        verify_files()
        
        print("🔧 Carregando tokenizer...")
        tokenizer = load_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        
        txt_files = [f for f in os.listdir(CONFIG['input_folder']) if f.lower().endswith('.txt')]
        if not txt_files:
            raise FileNotFoundError("Nenhum arquivo .txt encontrado na pasta de entrada.")
        
        print(f"📂 {len(txt_files)} arquivos TXT encontrados.")
        all_chunks = []
        
        for txt_file in txt_files:
            file_path = os.path.join(CONFIG['input_folder'], txt_file)
            print(f"\n📖 Processando: {txt_file}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            text = clean_text(text)
            chunks = chunk_text(text, tokenizer)
            print(f"  ➡️ {len(chunks)} chunks gerados.")
            all_chunks.extend(chunks)
        
        print(f"\n✍️ Salvando todos os {len(all_chunks)} chunks no JSONL...")
        os.makedirs(os.path.dirname(CONFIG['output_jsonl']), exist_ok=True)
        
        with open(CONFIG['output_jsonl'], 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(all_chunks):
                prompt = CONFIG['prompt_template'].format(text=chunk)
                if i % 3 == 0:
                    example = {
                        "text": "[INST] Exemplo de análise acadêmica [/INST] Aqui está um exemplo de resposta estruturada."
                    }
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
                entry = {"text": prompt}
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✅ Concluído! Dados salvos em: {CONFIG['output_jsonl']}")
        print(f"📊 Estatísticas:")
        print(f"- Total de chunks: {len(all_chunks)}")
        print(f"- Tamanho médio: {sum(len(c) for c in all_chunks)/len(all_chunks):.0f} caracteres por chunk")
    
    except Exception as e:
        print(f"❌ Erro no pré-processamento: {e}")

if __name__ == "__main__":
    main()
