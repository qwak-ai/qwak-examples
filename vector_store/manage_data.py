from typing import List

from sentence_transformers import SentenceTransformer


def create_embeddings(input_text: str) -> List[float]:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding = model.encode([input_text])
    return embedding.tolist()[0]
