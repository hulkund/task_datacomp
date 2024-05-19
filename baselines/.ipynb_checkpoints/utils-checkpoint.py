import numpy as np

def fix_embedding(embed):
    embedding = {}
    for e in embed:
        for key, value in e.items():
            if key not in embedding:
                embedding[key] = []
            embedding[key].append(value)
    return embedding