import os

from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embedder import create_embeddings
from rag.retriever import Retriever

from rag.index_manager import (
    save_index,
    load_index,
    save_chunks,
    load_chunks
)

def get_retriever(
        pdf_path,
        name):

    index_path = (
        f"storage/indexes/{name}.faiss"
    )

    chunk_path = (
        f"storage/chunks/{name}.pkl"
    )

    if (
        os.path.exists(index_path)
        and
        os.path.exists(chunk_path)
    ):

        index = load_index(
            index_path
        )

        chunks = load_chunks(
            chunk_path
        )

        return Retriever.from_saved_index(
            index,
            chunks
        )

    text = load_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = create_embeddings(
        chunks
    )

    retriever = Retriever(
        embeddings,
        chunks
    )

    save_index(
        retriever.index,
        index_path
    )

    save_chunks(
        chunks,
        chunk_path
    )

    return retriever