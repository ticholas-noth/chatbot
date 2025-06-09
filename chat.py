import os
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load OpenAI key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"  # or "gpt-3.5-turbo"

# Load data
print("🔄 Loading index and texts...")
index = faiss.read_index("index.faiss")
texts_df = pd.read_csv("texts.csv")

def get_embedding(text: str, model=EMBED_MODEL):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def ask_question(query, k=3):
    print(f"\n🔍 Searching for: {query}")
    query_embedding = np.array(get_embedding(query)).astype("float32")
    D, I = index.search(np.array([query_embedding]), k)
    
    # Retrieve top-k context texts
    contexts = texts_df.iloc[I[0]]["text"].tolist()
    context_str = "\n---\n".join(contexts)

    system_prompt = "You are a helpful assistant that answers questions about vacation rental properties using the provided context."
    user_prompt = f"""Answer the question using only the context below. If it's not in the context, say "I don't know."

Context:
{context_str}

Question: {query}
"""

    print("💬 Thinking...")
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("✅ Chat ready. Type your question and press Enter.")
    while True:
        try:
            q = input("\nYou: ")
            if q.lower() in ["exit", "quit"]:
                break
            answer = ask_question(q)
            print(f"\n🧠 Answer: {answer}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
