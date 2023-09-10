from uuid import uuid4

from qwak.exceptions import QwakException
from qwak.vector_store import VectorStoreClient

from vector_store.manage_collections import create_collection
from vector_store.manage_data import create_embeddings

if __name__ == '__main__':
    client = VectorStoreClient()

    # Create a collection
    collection = None
    collection_name = "product_catalog"

    try:
        # Retrieve the collection
        collection = client.get_collection_by_name(collection_name)
    except QwakException:
        collection = create_collection(client=client,
                                       name=collection_name,
                                       description="Indexing a product catalog of fashion items",
                                       dimension=384,
                                       metric="cosine",
                                       vectorizer="<optional>")

    # Get all existing collections
    collections = client.list_collections()

    # Upsert vectors
    vector_id = str(uuid4())
    collection.upsert(
        ids=[vector_id],
        vectors=[create_embeddings("Black sports shoes")],
        properties=[{
            "name": "shoes",
            "color": "black",
            "size": "42"
        }]
    )

    # Search vectors
    search_results = collection.search(output_properties=["name", "color", "size"],
                                       vector=create_embeddings("Black sport shoes"),
                                       top_results=5)
    # Delete the vector
    collection.delete(
        vector_ids=[vector_id]
    )

    # Delete the collection
    client.delete_collection(name=collection_name)
