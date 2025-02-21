import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load raw advice data
with open("data/kaggle/financial_advice.json", "r") as f:
    advice_data = json.load(f)

# Process and generate embeddings
embedded_advice = []
for item in advice_data:
    combined_text = f"{item['about_me']} {item['context']} {item['response']}"
    embedding = model.encode(combined_text).tolist()

    embedded_advice.append({
        "about_me": item["about_me"],
        "context": item["context"],
        "response": item["response"],
        "embedding": embedding
    })

# Save processed embeddings
with open("data/processed/embedded_advice.json", "w") as f:
    json.dump(embedded_advice, f, indent=4)

print(f"Processed {len(embedded_advice)} items and saved embeddings.")
