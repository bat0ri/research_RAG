from sentence_transformers import SentenceTransformer


LOCAL_EMBEDDER_MODEL = 'intfloat/multilingual-e5-small'

_embedder = SentenceTransformer(LOCAL_EMBEDDER_MODEL)

def encode_batch(batch):
    return _embedder.encode(batch).tolist()



