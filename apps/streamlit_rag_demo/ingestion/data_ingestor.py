from typing import List
from uuid import uuid4
from qwak.vector_store import VectorStoreClient


def chunk_text(text: str,
               max_chunk_size: int,
               overlap_size: int
               ) -> List[str]:
    """Helper function to chunk a text into overlapping chunks of specified size."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(text[start:end])
        start += max_chunk_size - overlap_size
    return chunks


def transform_record(record: str,
                     max_chunk_size: int = 500,
                     overlap_size: int = 100
                     ) -> List[dict]:
    """Transform a single record as described in the prompt."""

    chunks = chunk_text(record, max_chunk_size, overlap_size)
    transformed_records = []
    record_id = str(uuid4())
    for i, chunk in enumerate(chunks):
        chunk_id = f"{record_id}-{i+1}"
        transformed_records.append({
            'chunk_id': chunk_id,
            'chunk_parent_id': record_id,
            'chunk_text': chunk
        })
    return transformed_records


def insert_vector_data(chunks_array: List[dict],
                       collection_name: str):

    client = VectorStoreClient()
    collection = client.get_collection_by_name(collection_name)

    # Ingesting all the records in the vector store
    collection.upsert(
        ids=[
            str(uuid4()) for _ in range(len(chunks_array))
        ],
        natural_inputs=[
            c["chunk_text"] for c in chunks_array
        ],
        properties=chunks_array
    )


def insert_text_into_vector_store(input_path: str):
    """
    Inserts a file into the vector store
    :param input_path: The path of the file to be ingested
    """

    with open(input_path, 'r', encoding='ISO-8859-1') as f:
        contents = f.read()
        chunked_data = []
        chunk_array = transform_record(contents)
        for chunk in chunk_array:
            chunked_data.append(chunk)

        insert_vector_data(chunk_array, collection_name="financial-data")


if __name__ == '__main__':
    insert_text_into_vector_store(
        input_path='data/Tesla-10k.txt'
    )
