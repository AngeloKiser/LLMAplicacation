import os
import PyPDF2

def extract_pdf_text(pdf_path):
    """Extrai texto de um PDF de forma segura"""
    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            
            # Verifica criptografia
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception as e:
                    print(f"[AVISO] Erro de descriptografia: {e}")
                    return ""
            
            # Processa cada página
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        return text.strip()
    
    except Exception as e:
        print(f"[ERRO] Erro fatal: {e}")
        return ""

def save_text_to_file(text, output_path):
    """Salva texto em arquivo com tratamento de erros"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao salvar arquivo: {e}")
        return False

def process_pdf_folder(folder_path):
    """Processa todos os PDFs em uma pasta"""
    if not os.path.isdir(folder_path):
        print(f"[ERRO] Pasta não encontrada: {folder_path}")
        return
    
    # Lista todos os arquivos PDF na pasta
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("[INFO] Nenhum arquivo PDF encontrado na pasta.")
        return
    
    for pdf_file in pdf_files:
        full_pdf_path = os.path.join(folder_path, pdf_file)
        output_path = os.path.splitext(full_pdf_path)[0] + ".txt"
        
        print(f"\n[PROC] Processando: {pdf_file}")
        extracted_text = extract_pdf_text(full_pdf_path)
        
        if not extracted_text:
            print("[ERRO] Falha na extração de texto. O PDF pode ser baseado em imagens?")
            continue
        
        if save_text_to_file(extracted_text, output_path):
            print(f"[OK] Texto salvo em: {output_path}")
            with open(full_pdf_path, 'rb') as f:
                total_pages = len(PyPDF2.PdfReader(f).pages)
            print(f"Total de páginas: {total_pages}")
            print(f"Total de caracteres: {len(extracted_text)}")
        else:
            print("[ERRO] Falha ao salvar o arquivo.")

def main():
    """Função principal"""
    # Caminho da pasta fornecido pelo usuário
    folder_path = r"C:\Users\c0678507.sp\Documents\Mistral\data"  # Altere para o caminho da sua pasta
    
    process_pdf_folder(folder_path)

if __name__ == "__main__":
    main()