import numpy as np

def get_embedding(image):
    # Simulasi hasil ekstraksi wajah (128 dimensi)
    return np.random.rand(128)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Basis pengetahuan (Knowledge Base)
knowledge_base = [
    {"user": "user_001", "embedding": np.random.rand(128), "threshold": 0.8},
    {"user": "user_002", "embedding": np.random.rand(128), "threshold": 0.8},
]
