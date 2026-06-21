from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embedder import create_embeddings
from rag.retriever import Retriever
from utils.logger import logger

from vision.image_captioner import generate_caption

from llm.generator import generate_answer


def main():

    print("=" * 60)
    print("STEP 1 : Loading PDF")
    print("=" * 60)
    logger.info("Application Started")

    pdf_path = "data/papers/Federated_Learning_in_NTN_Design_Architecture_and_Challenges.pdf"

    text = load_pdf(pdf_path)
    logger.info("PDF loaded successfully")

    print(f"PDF Length: {len(text)} characters")

    print("\n")

    print("=" * 60)
    print("STEP 2 : Chunking")
    print("=" * 60)

    chunks = chunk_text(text)
    logger.info(
    f"Created {len(chunks)} chunks"
)

    print(f"Number of Chunks: {len(chunks)}")

    print("\n")

    print("=" * 60)
    print("STEP 3 : Creating Embeddings")
    print("=" * 60)

    embeddings = create_embeddings(chunks)
    logger.info(f"Generated embeddings with shape {embeddings.shape}")

    print(f"Embedding Shape: {embeddings.shape}")

    print("\n")

    print("=" * 60)
    print("STEP 4 : Building Retriever")
    print("=" * 60)

    retriever = Retriever(
        embeddings,
        chunks
    )
    logger.info("Retriever initialized")

    print("Retriever Ready")

    print("\n")

    print("=" * 60)
    print("STEP 5 : Question")
    print("=" * 60)

    question = "Explain the transformer architecture"
    logger.info(
    f"Question asked: {question}"
    )

    print(question)

    print("\n")

    print("=" * 60)
    print("STEP 6 : Retrieve Context")
    print("=" * 60)

    query_embedding = create_embeddings(
        [question]
    )

    results = retriever.retrieve(
        query_embedding
    )

    context = "\n".join(results)

    print(context[:1000])

    print("\n")

    print("=" * 60)
    print("STEP 7 : Image Captioning")
    print("=" * 60)

    image_path = "data/images/img_1.png"

    image_caption = generate_caption(
        image_path
    )
    logger.info(
    f"Image caption generated: {image_caption}"
    )

    print("Caption:")
    print(image_caption)

    print("\n")

    print("=" * 60)
    print("STEP 8 : Build Prompt")
    print("=" * 60)

    prompt = f"""
You are a research assistant.

Answer using the information below.

Question:
{question}

PDF Context:
{context}

Image Description:
{image_caption}

Provide:

1. Direct Answer
2. Supporting Evidence
3. Important Observations
"""

    print(prompt[:1000])

    print("\n")

    print("=" * 60)
    print("STEP 9 : LLM Response")
    print("=" * 60)
    logger.info(
    "Sending prompt to LLM"
  )

    answer = generate_answer(
        prompt
    )

    logger.info(
    "LLM response generated"
    )

    print(answer)

    print("\n")

    print("=" * 60)
    print("PROJECT TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()