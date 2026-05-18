from sentence_transformers import CrossEncoder


CROSS_ENCODER_MODEL = 'DiTy/cross-encoder-russian-msmarco'


class CrossEncoderReranker:

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list, metadatas: list,
               max_n: int = 5, min_n: int = 1, min_gap: float = 0.5):
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents, metadatas), key=lambda x: x[0], reverse=True)[:max_n]

        cut = len(ranked)
        if len(ranked) > min_n:
            score_vals = [s for s, _, _ in ranked]
            gaps = [score_vals[i] - score_vals[i + 1] for i in range(len(score_vals) - 1)]
            max_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
            # Only cut if the biggest gap is meaningful
            if gaps[max_gap_idx] >= min_gap:
                cut = max(max_gap_idx + 1, min_n)

        docs_out = [d for _, d, _ in ranked[:cut]]
        metas_out = [m for _, _, m in ranked[:cut]]
        scores_out = [s for s, _, _ in ranked[:cut]]
        return docs_out, metas_out, scores_out
