from typing import List
from uuid import uuid4
import tqdm
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
        chunk_id = f"{record_id}-{i + 1}"
        transformed_records.append({
            'chunk_id': chunk_id,
            'chunk_parent_id': record_id,
            'chunk_text': chunk
        })
    return transformed_records


def split_list_into_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]


def insert_vector_data(chunks_array: List[dict],
                       collection_name: str,
                       batch_size: int = 10):
    client = VectorStoreClient()
    collection = client.get_collection_by_name(collection_name)

    batches = list(split_list_into_batches(chunks_array, batch_size))
    batches = tqdm.tqdm(batches)

    # Ingesting all the records in the vector store
    for batch in batches:
        collection.upsert(
            ids=[
                str(uuid4()) for _ in range(len(batch))
            ],
            natural_inputs=[
                c["chunk_text"] for c in batch
            ],
            properties=batch
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

        insert_vector_data(chunks_array=chunk_array,
                           collection_name="financial-data")


if __name__ == '__main__':
    insert_text_into_vector_store(
        input_path='data/Tesla-10k.txt'
    )
