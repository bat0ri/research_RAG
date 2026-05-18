from chromadb.api.models import Collection
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class ChromaRetriever:
    collection: Collection
    model: SentenceTransformer = SentenceTransformer('intfloat/multilingual-e5-small')

    def __init__(self, collection):
        self.collection = collection

    def top_n(self, query: str, n: int = 3):
        embedding = self.model.encode(["query: " + query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n
        )
        return results


class BM25Retriever:

    def __init__(self, collection: Collection):
        all_docs = collection.get(include=["documents", "metadatas"])
        self.documents = all_docs["documents"]
        self.metadatas = all_docs["metadatas"]
        tokenized = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def top_n(self, query: str, n: int = 3):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]],
            "scores": [scores[i] for i in top_indices],
        }


class HybridRetriever:
    """Combines ChromaRetriever and BM25Retriever via Reciprocal Rank Fusion."""

    def __init__(self, collection: Collection, rrf_k: int = 60):
        self.semantic = ChromaRetriever(collection)
        self.bm25 = BM25Retriever(collection)
        self.rrf_k = rrf_k

    def top_n(self, query: str, n: int = 3, fetch_n: int = 20):
        semantic_results = self.semantic.top_n(query, n=fetch_n)
        bm25_results = self.bm25.top_n(query, n=fetch_n)

        semantic_docs = semantic_results["documents"][0]
        bm25_docs = bm25_results["documents"][0]

        # Build RRF score map keyed by document text
        rrf_scores: dict[str, float] = {}
        doc_to_meta: dict[str, dict] = {}

        for rank, doc in enumerate(semantic_docs):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_to_meta[doc] = semantic_results["metadatas"][0][rank]

        bm25_metas = bm25_results["metadatas"][0]
        for rank, doc in enumerate(bm25_docs):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if doc not in doc_to_meta:
                doc_to_meta[doc] = bm25_metas[rank]

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n]

        documents = [doc for doc, _ in ranked]
        metadatas = [doc_to_meta[doc] for doc, _ in ranked]
        scores = [score for _, score in ranked]

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "scores": scores,
        }
