import faiss
import pickle


def save_index(index, filepath):

    faiss.write_index(
        index,
        filepath
    )


def load_index(filepath):

    return faiss.read_index(
        filepath
    )


def save_chunks(chunks, filepath):

    with open(filepath, "wb") as f:

        pickle.dump(
            chunks,
            f
        )


def load_chunks(filepath):

    with open(filepath, "rb") as f:

        chunks = pickle.load(f)

    return chunks