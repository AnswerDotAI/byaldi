# Byaldi: Cross-Modal Semantic Text Retrieval

**A modified version of Byaldi that transforms image queries into semantically relevant text chunks using advanced multi-modal embedding techniques.**

This fork of the original Byaldi library reimagines multi-modal retrieval by enabling image-to-text semantic search. Departing from the original image-based embeddings, this version focuses on creating rich text embeddings and providing intelligent cross-modal retrieval capabilities.

Key Modification: Convert visual queries into meaningful text retrievals, bridging the gap between image and textual information spaces through innovative embedding strategies.

## Key Changes
- Switched from image-based embeddings to text-based embeddings
- Added support for image-to-text semantic retrieval
- Enables searching text chunks when given image queries

## Getting Started

### Loading a model

```python3
from byaldi import RAGMultiModalModel
RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v0.1")

```

### Creating an index 

```python3
from byaldi import RAGMultiModalModel
RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v0.1")
RAG.index(
        input_path="path/to/input/doc", # either text file or pdf file
        index_name="attention",
        overwrite=True
    )
```

### Searching

```python3
query = "image.jpg" # path to image for which you want to retrieve text chunks
results = RAG.search(query, k=1)
print(results)
```

