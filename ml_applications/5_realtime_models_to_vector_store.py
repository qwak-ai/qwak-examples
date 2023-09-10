from uuid import uuid4

from qwak.applications import QwakApplication, step
from qwak.exceptions import QwakException
from qwak_inference import RealTimeClient
from qwak.vector_store import VectorStoreClient

from typing import List
from sentence_transformers import SentenceTransformer


def create_embeddings(input_text: str) -> List[float]:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding = model.encode([input_text])
    return embedding.tolist()[0]


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

class RealTimeModelToVectorStoreApp(QwakApplication):
    """
    1. Run the real time model
    2. Save the model output in the vector store
    """
    @step
    def start(self):
        self.next(self.run_realtime_model)

    @step
    def run_realtime_model(self, params):
        df = params["df"]
        client = RealTimeClient(model_id="sentence_embeddings_aecc6f")
        embedding = client.predict(df)
        self.next(self.run_vector_store_ingestion,
                  params={
                      "embedding": embedding
                  })

    @step
    def run_vector_store_ingestion(self, params):

        embedding = params["embedding"]
        properties = params["properties"]

        client = VectorStoreClient()
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

        # Upsert vectors
        vector_id = str(uuid4())
        collection.upsert(
            ids=[vector_id],
            vectors=[embedding],
            properties=[properties]
        )


if __name__ == '__main__':
    flow = RealTimeModelToVectorStoreApp()
    flow.execute(schedule="0 * * * *")
    flow.execute_now()
