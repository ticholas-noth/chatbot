from flask_cors import CORS
from flask import Flask, request, jsonify
import faiss
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

# openai api recommends the api key to be an env variable
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)  # Allow all origins by default for API


# Load index and texts generated from embed_data.py
index = faiss.read_index("index.faiss")
texts = pd.read_csv("texts.csv")

EMBED_MODEL = "text-embedding-3-small"

def get_embedding(text):
    response = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return response.data[0].embedding

def ask_question(question):
    embedding = np.array(get_embedding(question)).astype("float32")
    D, I = index.search(np.array([embedding]), k=3)
    results = [texts.iloc[i]["text"] for i in I[0]]

    prompt = f"""
            You are a helpful assistant for vacation rentals.
            Here is some property data:

            {results[0]}

            Answer this question:
            {question}
            """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about vacation rental properties using the provided context. Answer the question using only the context below. If it's not in the context, say I dont know, ask me about this property! If the question is about the property, but not in the context, say I am unsure of that. Please call 123-456-7890 for more information regarding your question."},
            {"role": "user", "content": prompt}
        ],
        # value 0-2 that determines the focus level of the response. Higher number = more random. Lower number = deterministic
        temperature=0.3
    )

    return response.choices[0].message.content

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = ask_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
