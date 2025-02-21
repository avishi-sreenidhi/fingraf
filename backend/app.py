from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load retrieval model
retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)
CORS(app)

# Load embedded advice dataset
with open('data/processed/embedded_advice.json', 'r') as f:
    embedded_advice = json.load(f)

# Extract embeddings and metadata
embeddings = np.array([item['embedding'] for item in embedded_advice])
advice_metadata = [
    {'about_me': item['about_me'], 'context': item['context'], 'response': item['response']}
    for item in embedded_advice
]

def find_best_matches(query, top_k=3, min_similarity=0.3):
    query_embedding = retrieval_model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    best_indices = similarities.argsort()[-top_k:][::-1]
    
    best_matches = [(advice_metadata[idx], similarities[idx]) for idx in best_indices if similarities[idx] >= min_similarity]
    
    return best_matches

def filter_context_data(retrieved_contexts):
    """
    Remove 'age' and 'occupation' fields from matched contexts before sending them to the LLM.
    """
    filtered_contexts = []
    for ctx in retrieved_contexts:
        filtered_entry = {k: v for k, v in ctx[0].items() if k not in ["age", "occupation"]}
        filtered_contexts.append((filtered_entry, ctx[1]))  # Keep similarity score
    
    return filtered_contexts

def process_query(query):
    retrieved_contexts = find_best_matches(query, top_k=10)
    
    # Filter out unwanted fields
    filtered_contexts = filter_context_data(retrieved_contexts)

    final_context = "\n".join([
        f"About Me: {ctx[0].get('about_me', '')}\nContext: {ctx[0].get('context', '')}\nResponse: {ctx[0].get('response', '')}"
        for ctx in filtered_contexts
    ])

    return final_context, filtered_contexts

@app.route("/")
def home():
    return jsonify({"message": "Welcome to FinGraphAI!"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/query", methods=["POST"])
def get_response():
    return handle_query()

def handle_query():
    data = request.get_json()
    print("Received request:", data)  # Debugging line
    
    user_query = data.get("query")  # Ensure correct key
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    final_context, retrieved_contexts = process_query(user_query)
    print("Final context:", final_context)  # Debugging line

    overall_query = f"{final_context}\nUser Query: {user_query}"
    print("Overall query:", overall_query)  # Debugging line

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": overall_query}
            ],
            max_tokens=150,
            temperature=0.7
        )
        llm_response = response.choices[0].message.content.strip()
    except Exception as e:
        print("Error with OpenAI response:", e)
        llm_response = "Sorry, I couldn't generate a response."

    return jsonify({
        "matches": retrieved_contexts,
        "overall_query": overall_query,
        "llm_response": llm_response
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
