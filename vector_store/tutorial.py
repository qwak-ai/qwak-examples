from uuid import uuid4

from qwak.exceptions import QwakException
from qwak.vector_store import VectorStoreClient
from qwak.vector_store.filters import GreaterThan

if __name__ == '__main__':
    """
    This code snippet creates a collection, stores natural input data as a vector, searches the collection
    and finally deletes it
    """

    client = VectorStoreClient()

    collection = None
    collection_name = "product_catalog"

    # Retrieve a collection or create a new one
    try:
        collection = client.get_collection_by_name(collection_name)
    except QwakException:
        collection = client.create_collection(
            name=collection_name,
            description="Indexing a product catalog of fashion items",
            dimension=384,
            metric="cosine",
            vectorizer="Sentence Embeddings aecc6f"  # The name of a deployed realtime model on Qwak
        )

    # Get all existing collections
    collections = client.list_collections()

    # Add a new vector to the collection
    vector_id = str(uuid4())
    metadata = {
        "name": "shoes",
        "color": "black",
        "size": 42,
        "price": 19.99
    }
    collection.upsert(
        ids=[vector_id],
        natural_inputs=["Black sports shoes"],
        properties=[metadata]
    )

    # Search all vectors
    search_results = collection.search(
        output_properties=list(metadata.keys()),
        natural_input="Black sport shoes",
        top_results=5
    )

    # Search filtered properties
    # Available filters: And, Or, Equal, Filter, GreaterThan, GreaterThanEqual,
    #                    IsNotNull, IsNull, LessThan, LessThanEqual, Like, NotEqual
    filtered_search_results = collection.search(
        output_properties=list(metadata.keys()),
        natural_input="Black sport shoes",
        filter=GreaterThan(property="size", value=10.99)
    )
    print(filtered_search_results)

    empty_filtered_search_results = collection.search(
        output_properties=list(metadata.keys()),
        natural_input="Black sport shoes",
        filter=GreaterThan(property="size", value=39.99)
    )
    print(empty_filtered_search_results)

    # Delete the vector
    collection.delete(
        vector_ids=[vector_id]
    )

    # Delete the collection
    client.delete_collection(name=collection_name)
