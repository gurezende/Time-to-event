from fastembed import TextEmbedding
from typing import List
import numpy as np
import json

documents = json(open("C:/MyDocuments/survival/scripts/churned_customers.json", "r"))

embedding_model = TextEmbedding()
print("The model BAAI/bge-small-en-v1.5 is ready to use.")

embeddings_generator = embedding_model.embed(documents)
embeddings_list = list(embeddings_generator)

# Qdrant client
from qdrant_client import QdrantClient, models

model_name = "BAAI/bge-small-en"

# Encode the text: embeddings
vectors = embedding_model.encode(
    documents.tolist(),
    show_progress_bar=True,
)

client = QdrantClient(":memory:") 
client.create_collection(
    collection_name="test_collection",
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name), 
        distance=models.Distance.COSINE
    ),  # size and distance are model dependent
)

client.upload_collection(
    collection_name="test_collection",
    vectors=embeddings_list
)

search_result = client.query_points(
    collection_name="test_collection",
    query=models.Document(
        text="Show me the product description", 
        model=model_name
    )
).points
print(search_result)