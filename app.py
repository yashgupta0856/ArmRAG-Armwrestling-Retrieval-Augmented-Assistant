from flask import Flask, request, jsonify, render_template
from rag.loader import load_file, split_into_chunks
from rag.vectorstore import build_vectorstore
from rag.rag_pipeline import rag

app = Flask(__name__)

pages = load_file("pdfContent/document.pdf")
chunks, metas = split_into_chunks(pages)
build_vectorstore(chunks, metas)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    answer, sources = rag(question)
    return jsonify({"answer": answer, "sources": sources})
