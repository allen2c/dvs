# DVS - DuckDB Vector Similarity Search

[![PyPI version](https://badge.fury.io/py/dvs-py.svg)](https://badge.fury.io/py/dvs-py)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for vector similarity search powered by DuckDB and OpenAI embeddings.

## Features

- **Fast Vector Search**: Efficient similarity search using DuckDB's vector capabilities
- **OpenAI Integration**: Automatic embedding generation with OpenAI models
- **Caching**: Built-in embedding cache for improved performance
- **Simple API**: Easy-to-use Python interface
- **Flexible Storage**: Store documents with metadata

## Installation

```bash
pip install dvs-py
```

## Quick Start

### Basic Usage

```python
import asyncio
import tempfile
import openai_embeddings_model as oai_emb_model
from dvs import DVS

# Initialize DVS with a database file and model
dvs = DVS(
    tempfile.NamedTemporaryFile(suffix=".duckdb").name,
    model="text-embedding-3-small",
    model_settings=oai_emb_model.ModelSettings(dimensions=1536)
)

# Add documents
dvs.add("Apple announced new iPhone features with upgraded camera and A16 chip.")
dvs.add("Microsoft updated Azure with enhanced AI tools and security features.")

# Search
results = asyncio.run(dvs.search("What are the new iPhone features?"))
print(f"Found {len(results)} results")
for point, document, score in results:
    print(f"Score: {score:.3f} - {document.content[:100]}...")
```

### Advanced Configuration

```python
import asyncio
import pathlib
import diskcache
import openai
import openai_embeddings_model as oai_emb_model
from dvs import DVS

# Configure with custom cache and model settings
dvs = DVS(
    "./my_database.duckdb",
    model=oai_emb_model.OpenAIEmbeddingsModel(
        model="text-embedding-3-small",
        openai_client=openai.OpenAI(),
        cache=diskcache.Cache("./cache/embeddings.cache"),
    ),
    model_settings=oai_emb_model.ModelSettings(dimensions=1536),
    verbose=True
)

# Add documents with metadata
from dvs.types.document import Document

doc = Document.from_content(
    "Latest developments in artificial intelligence...",
    name="AI Research Paper",
    metadata={"author": "John Doe", "year": 2024}
)
dvs.add(doc)

# Search with more results
results = asyncio.run(dvs.search("artificial intelligence", top_k=10))
```

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Document Management

### Adding Documents

```python
# Add single document
dvs.add("Your document content here")

# Add multiple documents
documents = [
    "First document content",
    "Second document content",
    "Third document content"
]
dvs.add(documents)

# Add documents with metadata
from dvs.types.document import Document

docs = [
    Document.from_content("Content 1", name="Doc 1", metadata={"category": "tech"}),
    Document.from_content("Content 2", name="Doc 2", metadata={"category": "science"})
]
dvs.add(docs)
```

### Searching Documents

```python
# Basic search
results = asyncio.run(dvs.search("your query"))

# Search with more results
results = asyncio.run(dvs.search("your query", top_k=10))

# Search with embeddings included
results = asyncio.run(dvs.search("your query", with_embedding=True))
```

### Removing Documents

```python
# Get document ID from search results
results = asyncio.run(dvs.search("some query"))
doc_id = results[0][1].document_id

# Remove document
dvs.remove(doc_id)

# Remove multiple documents
dvs.remove([doc_id1, doc_id2, doc_id3])
```

## Development

Install development dependencies:

```bash
make install-all
```

Run tests:

```bash
make pytest
```

Format code:

```bash
make format-all
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/allen2c/dvs/issues) on GitHub.
