#!/usr/bin/env python

import os

# Embedding model config
FASTEMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Qdrant config
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "blog-content")
QDRANT_LOCATION = os.getenv("QDRANT_LOCATION", "http://localhost:6333")
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", 32))

# Langchain config
LANGCHAIN_RAG_PROMPT = os.getenv("LANGCHAIN_RAG_PROMPT", "rlm/rag-prompt")

# LLM config

# Either use a custom endpoint or specify a model to use from Huggingface
HUGGINGFACE_CUSTOM_ENDPOINT = os.getenv("HUGGINGFACE_CUSTOM_ENDPOINT")

HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Document loader config
BLOG_URL = os.getenv("BLOG_URL", "https://lilianweng.github.io/posts/2023-06-23-agent/")

# Text splitter config
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
