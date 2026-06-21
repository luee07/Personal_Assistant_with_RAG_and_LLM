from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embedder import create_embeddings
from rag.retriever import Retriever

from vision.image_captioner import (
    generate_caption
)

def build_pdf_index(pdf_path):

    text = load_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = create_embeddings(
        chunks
    )

    retriever = Retriever(
        embeddings,
        chunks
    )

    return retriever

def retrieve_context(
        retriever,
        question):

    query_embedding = create_embeddings(
        [question]
    )

    results = retriever.search(
        query_embedding
    )

    return "\n".join(results)

def analyze_image(image_path):

    caption = generate_caption(
        image_path
    )

    return caption

def build_prompt(
        question,
        pdf_context,
        image_caption):

    return f"""
You are a research assistant.

Answer using ONLY the information
provided.

Question:
{question}

PDF Context:
{pdf_context}

Image Description:
{image_caption}

Provide:

1. Direct Answer
2. Supporting Evidence
3. Important Observations

If information is missing,
say so.
"""

    return prompt


from llm.generator import (
    generate_answer
)

def run_pipeline(
        pdf_path,
        image_path,
        question):

    retriever = build_pdf_index(
        pdf_path
    )

    pdf_context = retrieve_context(
        retriever,
        question
    )

    image_caption = analyze_image(
        image_path
    )

    prompt = build_prompt(
        question,
        pdf_context,
        image_caption
    )

    answer = generate_answer(
        prompt
    )

    return answer