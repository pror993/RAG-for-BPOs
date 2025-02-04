from sentence_transformers import SentenceTransformer
import os
import json
from typing import List, Dict

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Hugging Face embedding model.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of input texts.
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def process_chunks(self, chunk_dir: str) -> List[Dict[str, List[float]]]:
        """
        Generate embeddings for all text chunks in a directory.
        """
        embeddings = []
        for chunk_file in os.listdir(chunk_dir):
            if chunk_file.endswith(".txt"):
                with open(os.path.join(chunk_dir, chunk_file), "r") as f:
                    text = f.read()
                    embedding = self.generate_embeddings([text])[0]
                    embeddings.append({"chunk_file": chunk_file, "embedding": embedding})
        return embeddings

    def save_embeddings(self, embeddings: List[Dict[str, List[float]]], output_file: str):
        """
        Save embeddings to a JSON file.
        """
        with open(output_file, "w") as f:
            json.dump(embeddings, f)

# Example Usage
if __name__ == "__main__":
    chunk_dir = "./processed_chunks/"
    output_file = "./embeddings/embeddings.json"
    generator = EmbeddingGenerator()
    embeddings = generator.process_chunks(chunk_dir)
    generator.save_embeddings(embeddings, output_file)
