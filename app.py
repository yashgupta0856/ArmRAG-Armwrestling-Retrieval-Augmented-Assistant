import os
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify, render_template
from rag.loader import load_file, split_into_chunks
from rag.vectorstore import build_store
from rag.rag_pipeline import rag

app = Flask(__name__)

pages = load_file("pdfContent/document.pdf")
chunks, metas = split_into_chunks(pages)
build_store(chunks, metas)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question")
    answer, sources = rag(q)
    return jsonify({"answer": answer, "sources": sources})
