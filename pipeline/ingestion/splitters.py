from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_core.documents import Document


EMBEDDER_MODEL = 'intfloat/multilingual-e5-small'

SPLITTER_NAMES = ["character", "recursive_character", "token", "sentence_transformers"]
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def get_splitter(name: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    if name == "character":
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif name == "recursive_character":
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif name == "token":
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif name == "sentence_transformers":
        return SentenceTransformersTokenTextSplitter(
            model_name=EMBEDDER_MODEL,
            tokens_per_chunk=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"Unknown splitter '{name}'. Available: {SPLITTER_NAMES}")


def datasets_to_documents(dataset):
    return [
        Document(page_content=item['text'], metadata={"id": item["id"]})
        for item in dataset["train"]
    ]


def split_documents(dataset, splitter):
    return splitter.split_documents(datasets_to_documents(dataset))
