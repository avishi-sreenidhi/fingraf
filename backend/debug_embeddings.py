import os
import numpy as np

embedding_path = "backend/embeddings.npy"  # Change this to your actual path
if os.path.exists(embedding_path):
    embeddings = np.load(embedding_path)
    print(f"Embeddings shape: {embeddings.shape}")
else:
    print("Embeddings file not found!")
