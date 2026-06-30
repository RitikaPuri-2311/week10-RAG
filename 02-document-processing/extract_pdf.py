import fitz

def extract_pdf_text(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    pages_data = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages_data.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text)
        })
    doc.close()
    return pages_data

def print_extraction_summary(pages_data: list[dict]):
    total_chars = sum(p["char_count"] for p in pages_data)
    print(f"Total pages extracted : {len(pages_data)}")
    print(f"Total characters      : {total_chars}")
    print(f"\n--- Page 1 preview (first 300 chars) ---")
    print(pages_data[0]["text"][:300])

if __name__ == "__main__":
    file_path = "test-files/sample.pdf"
    pages = extract_pdf_text(file_path)
    print_extraction_summary(pages)