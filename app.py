from flask import Flask, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ArmRAG Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }
        .box {
            background: #020617;
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            max-width: 600px;
        }
        a {
            color: #38bdf8;
            font-size: 20px;
            text-decoration: none;
            font-weight: bold;
        }
        a:hover {
            text-decoration: underline;
        }
        h1 {
            margin-bottom: 16px;
        }
        p {
            color: #94a3b8;
        }
    </style>
</head>
<body>
    <div class="box">
        <h1> ArmRAG — Armwrestling RAG Project</h1>
        <p>This is a lightweight landing page.</p>
        <p>
             <a href="https://armwrestlingrag.streamlit.app/" target="_blank">
                Click here to open the full RAG application
            </a>
        </p>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

if __name__ == "__main__":
    app.run()
