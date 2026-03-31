from dotenv import load_dotenv
import os
import math

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer


load_dotenv()


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def get_top_k_chunks(question, chunks, embed_model, top_k=3):
    chunk_embeddings = embed_model.encode(chunks, convert_to_tensor=False)
    question_embedding = embed_model.encode([question], convert_to_tensor=False)[0]

    scored_chunks = []
    for chunk, chunk_emb in zip(chunks, chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_emb)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
    return top_chunks


def ask_groq(question, context):
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF question-answering assistant.

Rules:
1. Answer ONLY from the provided PDF context.
2. If the answer is not clearly available in the context, say:
   "I could not find the exact answer in the PDF."
3. Keep the answer clear and concise.

PDF Context:
{context}

Question:
{question}
""")

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


def main():
    pdf_path = input("Enter PDF file path: ").strip()

    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    if not text.strip():
        print("No text found in PDF.")
        return

    print("\nProcessing PDF...")
    chunks = split_text(text)

    print("Loading local embedding model (first run may take time)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nPDF Agent is ready! Ask questions from your PDF.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        top_chunks = get_top_k_chunks(question, chunks, embed_model, top_k=3)
        context = "\n\n".join(top_chunks)

        try:
            answer = ask_groq(question, context)
            print("\nAnswer:")
            print(answer)
            print("-" * 80)
        except Exception as e:
            print(f"\nGroq Error: {e}")
            print("-" * 80)


if __name__ == "__main__":
    main()