from uuid import uuid4

from pipeline.ingestion.splitters import get_splitter, split_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from pipeline.embedder import encode_batch


CHROMA_PATH: str = "chroma_db"


def insert_data(
    dataset,
    collection,
    split_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 1000,
) -> int:
    splitter = get_splitter(split_type, chunk_size, chunk_overlap)
    splits = split_documents(dataset, splitter)

    uuids = [str(uuid4()) for _ in range(len(splits))]

    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        batch_texts = ["passage: " + doc.page_content for doc in batch]
        collection.add(
            documents=batch_texts,
            embeddings=encode_batch(batch_texts),
            metadatas=[doc.metadata for doc in batch],
            ids=uuids[i:i + batch_size],
        )

    print(f"Добавлено {len(splits)} чанков в коллекцию '{collection.name}'")
    return len(splits)
