from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Load PDF
reader = PdfReader("pdfContent/document.pdf")
docs = []

for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    docs.append(
        Document(
            page_content=text,
            metadata={"page": i + 1}
        )
    )

# Load embedding model (LOCAL MACHINE ONLY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build FAISS index
vectorstore = FAISS.from_documents(docs, model)

# Save vectors
vectorstore.save_local("vectors")

print(" FAISS vectors saved in ./vectors")
