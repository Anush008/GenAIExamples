#!/usr/bin/env python

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from rag_qdrant.config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_LOCATION,
    LANGCHAIN_RAG_PROMPT,
    FASTEMBED_MODEL,
    HUGGINGFACE_MODEL,
    HUGGINGFACE_CUSTOM_ENDPOINT,
    HUGGINGFACEHUB_API_TOKEN
)

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from langchain import hub

assert any([HUGGINGFACE_MODEL, HUGGINGFACE_CUSTOM_ENDPOINT]), "Set either HUGGINGFACE_MODEL or HUGGINGFACE_CUSTOM_ENDPOINT in the environment variables"

if not HUGGINGFACE_CUSTOM_ENDPOINT:
    assert HUGGINGFACEHUB_API_TOKEN, "HUGGINGFACEHUB_API_TOKEN env is required when using a Huggingface model"

# Init Embeddings
embeddings = FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)

# Connect to pre-loaded vectorstore
# run the ingest.py script to populate this
client = QdrantClient(location=QDRANT_LOCATION)

vectorstore = Qdrant(collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings, client=client)


retriever = vectorstore.as_retriever()

prompt = hub.pull(LANGCHAIN_RAG_PROMPT)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_MODEL,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    streaming=True,
    truncate=1024,
)

rag_chain = (
    RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
