import os
import json
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"

def get_embedding(text: str, model=EMBED_MODEL):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def load_data(prop_path, amenity_path):
    with open(prop_path, "r", encoding="utf-8") as f:
        properties = json.load(f)
    with open(amenity_path, "r", encoding="utf-8") as f:
        amenities = json.load(f)

    # Group amenities by unit_id
    amenity_map = {}
    for item in amenities:
        unit = item["unit_id"]
        if unit not in amenity_map:
            amenity_map[unit] = []
        amenity_name = item.get("name", "")
        amenity_desc = item.get("description", "").strip()
        if amenity_desc and amenity_desc.lower() != "yes":
            full = f"{amenity_name}: {amenity_desc}"
        else:
            full = amenity_name
        amenity_map[unit].append(full)

    combined_data = []
    for prop in properties:
        unit_id = prop["unit_id"]
        text = f"Name: {prop['prop_name']}\n"
        text += f"Address: {prop['address']}, {prop['city']}, {prop['state']} {prop['zip']}\n"
        text += f"Beds: {prop['bed']} | Baths: {prop['bath']} | Sleeps: {prop['sleeps']}\n"

        description = clean_html(prop.get("description", ""))
        if description:
            text += f"Description: {description}\n"

        amenities_list = amenity_map.get(unit_id, [])
        if amenities_list:
            text += "Amenities: " + ", ".join(amenities_list)

        combined_data.append({"unit_id": unit_id, "text": text})

    return combined_data

def build_index(data):
    texts = [item["text"] for item in data]
    embeddings = [get_embedding(text) for text in texts]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    return index, data

if __name__ == "__main__":
    combined = load_data("data/vunique-pd.json", "data/vunique-amenities.json")
    index, data = build_index(combined)
    faiss.write_index(index, "index.faiss")
    pd.DataFrame(data).to_csv("texts.csv", index=False)
    print("✅ Embedding and indexing complete.")
