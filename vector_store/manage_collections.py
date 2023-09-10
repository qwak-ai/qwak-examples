from qwak.vector_store import VectorStoreClient


def create_collection(client: VectorStoreClient,
                      name: str,
                      description: str,
                      dimension: int,
                      metric: str,
                      vectorizer: str):
    return client.create_collection(name=name,
                                    description=description,
                                    dimension=dimension,
                                    metric=metric,
                                    vectorizer=vectorizer)
