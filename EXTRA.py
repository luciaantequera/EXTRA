# generar_embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Frases a transformar
corpus = [
    "La programación dinámica es una técnica de optimización.",
    "El algoritmo de Dijkstra encuentra el camino más corto en un grafo.",
    "Los árboles binarios de búsqueda permiten búsquedas eficientes.",
    "Los grafos dirigidos pueden tener ciclos.",
    "Python es un lenguaje de programación popular para inteligencia artificial."
]

# Cargar modelo preentrenado
modelo = SentenceTransformer('all-MiniLM-L6-v2')

# Generar embeddings
corpus_embeddings = modelo.encode(corpus)

# Ver forma de los embeddings
print(f"Forma de los embeddings: {corpus_embeddings.shape}")

# Guardar embeddings en archivo
np.save("embeddings.npy", corpus_embeddings)

# usar_embeddings.py

import numpy as np
from numpy import dot
from numpy.linalg import norm

# Cargar embeddings desde archivo
embeddings_cargados = np.load("embeddings.npy")
print(f"Embeddings cargados: {embeddings_cargados.shape}")

# Función de similitud coseno
def similitud_coseno(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Comparar la primera y la segunda frase
similitud = similitud_coseno(embeddings_cargados[0], embeddings_cargados[1])
print(f"Similitud entre la frase 1 y la frase 2: {similitud}")

##Embeddings cargados: (5, 384)
##Similitud entre la frase 1 y la frase 2: 0.3845...
