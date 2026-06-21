import faiss
import numpy as np

class Retriever:

    def __init__(self,
                 embeddings,
                 chunks):

        self.chunks = chunks

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings.astype(np.float32)
        )


    def retrieve(self,
                 query_embedding,
                 top_k=5):

        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            top_k
        )

        retrieved_chunks = [self.chunks[i] for i in indices[0]]

        return retrieved_chunks
    
    @classmethod
    def from_saved_index(
        cls,
        index,
        chunks):

        obj = cls.__new__(cls)

        obj.index = index

        obj.chunks = chunks

        return obj