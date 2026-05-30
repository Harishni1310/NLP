
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

text = '''
AI is transforming education,
RAG helps reduce hallucination.
Embeddings convert text into vectors.
'''
chunks = text.split('.')#spliting the data on the basis of fullstop

chunks = [chunk.strip() for chunk in chunks if chunk.strip()]


model = SentenceTransformer('all-MiniLM-l6-v2')
sentence = "AI is transforming education"
embedding = model.encode(sentence)

chunks=[
    "AI is transforming education",
    "RAG helps reduce hallucination",
    "Embedding convert text into vectors"
]
embedding=model.encode(chunks)

query = "What reduces hallucination?"
print(query)
query_embedding = model.encode(query)



similarities = cosine_similarity([query_embedding], embedding)

best_match_index = similarities.argmax()
print(chunks[best_match_index])