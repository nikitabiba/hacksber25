def load_faiss(documents, embeddings):
    import faiss
    return FAISS.from_documents(documents, embeddings)
