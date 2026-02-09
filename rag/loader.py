from PyPDF2 import PdfReader

def load_file(path):
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})

    return pages


def split_into_chunks(pages, chunk_size=1000, overlap=200):
    chunks, metadatas = [], []
    global_chunk_index = 0

    for p in pages:
        text = p["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            meta = {
                "page": p["page"],
                "start_char": start,
                "end_char": min(end, len(text)),
                "chunk_id": f"p{p['page']}-c{global_chunk_index}"
            }

            chunks.append(chunk)
            metadatas.append(meta)

            global_chunk_index += 1
            start = end - overlap

    return chunks, metadatas
