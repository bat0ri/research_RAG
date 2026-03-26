from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from uuid import uuid4




def insert_dataset(
    dataset,
    collection,
    model,
    chunk_size=500,
    chunk_overlap=100,
    batch_size=1000,
):
    
    documents = [
        Document(
            page_content=f"{item['text']}", 
            metadata={"id": item["id"]}
        )
        for i, item in enumerate(dataset["train"])
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)

    uuids = [str(uuid4()) for _ in range(len(splits))]
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        batch_ids = uuids[i:i+batch_size]
    
        batch_texts = ["passage: " + doc.page_content for doc in batch]
        embeddings = model.encode(batch_texts).tolist()

        chroma_docs = [doc.page_content for doc in batch]
        chroma_metadatas = [doc.metadata for doc in batch]

        collection.add(
            documents=chroma_docs,
            embeddings=embeddings,
            metadatas=chroma_metadatas,
            ids=batch_ids
        )
        
        print(f"Added batch {i//batch_size + 1}, size: {len(batch)}")
    
    print(f"Всего добавлено {len(splits)} чанков в коллекцию {collection.name}")
