import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
