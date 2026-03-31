# AI-Pdf-Chatbot
A lightweight Retrieval-Augmented Generation (RAG) pipeline for PDF Q&amp;A. Extracts and chunks PDF text, generates embeddings locally via sentence-transformers (all-MiniLM-L6-v2), retrieves top-k relevant chunks using cosine similarity, and answers questions using Groq's LLaMA 3.1 8B — all without a vector database.
