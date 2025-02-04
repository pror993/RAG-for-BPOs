from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        self.model_name = os.getenv("HF_EMBEDDING_MODEL")  # Dynamically load model name
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
