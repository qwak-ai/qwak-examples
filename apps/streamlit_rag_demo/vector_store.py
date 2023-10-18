from typing import List

from qwak.vector_store import VectorStoreClient


def retrieve_vector_context(query: str,
                            output_properties: List[str],
                            collection_name: str = "articles",
                            top_results: int = 2
                            ) -> str:
    """
    Retrieve the context from the Vector Store

    :param query: User query
    :param output_properties: Metadata fields to return with the query
    :param collection_name: Name of the vector store collection
    :param top_results: The number of results to return

    :return: The full context from the Vector store
    """
    client = VectorStoreClient()
    collection = client.get_collection_by_name(collection_name)

    vector_results = collection.search(
        natural_input=query,
        top_results=top_results,
        output_properties=output_properties
    )

    contexts = [
        x.properties["text"] for x in vector_results
    ]

    vector_contexts = (
        "\n\n---\n\n".join(contexts)
    )
    return vector_contexts
