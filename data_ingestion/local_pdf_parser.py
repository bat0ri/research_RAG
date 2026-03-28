import os
from pathlib import Path
from pdftext.extraction import plain_text_output


def parse_all_pdfs(input_dir: str, output_dir: str):
    """
        input_dir: папка с PDF файлами (datasets/docs/)
        output_dir: папка для сохранения txt (datasets/parsed_docs/)
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"PDF файлы не найдены в {input_dir}")
        return
    
    for pdf_path in pdf_files:
        try:
            print(f"Обработка: {pdf_path.name}...")
            text = plain_text_output(
                str(pdf_path),
                sort=True,
                hyphens=False
            )
            
            output_path = Path(output_dir) / f"parsed {pdf_path.stem}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"Сохранено")
            
        except Exception as e:
            print(f"Ошибка при обработке {pdf_path.name}: {e}")
    

if __name__ == "__main__":
    parse_all_pdfs("../datasets/docs", "../datasets/parsed_docs")