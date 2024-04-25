#!/usr/bin/env python

from bs4 import SoupStrainer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import WebBaseLoader
from rag_qdrant.config import (
    QDRANT_LOCATION,
    QDRANT_COLLECTION_NAME,
    FASTEMBED_MODEL,
    BLOG_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QDRANT_UPSERT_BATCH_SIZE
)

def ingest_blog():
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_path=BLOG_URL,
        bs_kwargs=dict(parse_only=SoupStrainer(class_=("post-content", "post-title", "post-header"))),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)

    embedding = FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)

    # Batch size
    batch_size = QDRANT_UPSERT_BATCH_SIZE

    num_chunks = len(splits)
    for i in range(0, num_chunks, batch_size):
        batch_chunks = splits[i : i + batch_size]

        Qdrant.from_documents(
            documents=batch_chunks,
            embedding=embedding,
            location=QDRANT_LOCATION,
            collection_name=QDRANT_COLLECTION_NAME,
        )
        print(f"Processed batch {i//batch_size + 1}/{(num_chunks-1)//batch_size + 1}")


if __name__ == "__main__":
    ingest_blog()
