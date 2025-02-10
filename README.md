# Associative Neural Memory DB

A next‑generation AI database engine inspired by human memory storage. This project implements a self‑contained, production‑ready engine that stores data as distributed, overlapping neural embeddings and provides a natural language query interface for associative (content‑addressable) retrieval.

> **Disclaimer:**  
> This prototype uses a simple character‑level neural embedding model built with PyTorch and heuristic natural language filter extraction. In a production environment, you would replace these with a pre‑trained or fine‑tuned embedding model and persistent storage.

## Features

- **Neural Embedding Model:**  
  A built‑in character‑level neural network (using PyTorch) that converts input text into fixed‑size embeddings.
  
- **Associative Memory Storage:**  
  Records are stored in‑memory with free‑form text, metadata (e.g., timestamps, monetary amounts), and neural embeddings.

- **Natural Language Query Interface:**  
  Queries are processed using both neural embeddings (for similarity matching via cosine similarity) and heuristic filters (e.g., "above $100" or "last week").

- **RESTful API with FastAPI:**  
  The engine is exposed via a fully documented REST API, including endpoints for record creation, bulk insertion, retrieval, update, deletion, and querying. The API documentation is automatically generated and accessible via Swagger UI.

- **Production‑Ready Structure:**  
  Designed to serve as a foundation for a new‑generation AI‑driven database. Easily extensible for persistent storage, improved embedding models, and more sophisticated natural language processing.

## Architecture Overview

1. **Neural Embedding Module:**  
   - Implements a simple character‑level feed‑forward network using PyTorch.
   - Converts text (tokenized at the character level) into a fixed‑dimensional embedding vector.

2. **Associative Memory Engine:**  
   - Stores records as dictionaries containing text, metadata, and computed embeddings.
   - Provides methods for inserting, updating, deleting, and querying records.

3. **Query Processing:**  
   - Converts natural language queries into embeddings.
   - Computes cosine similarity between query embeddings and stored record embeddings.
   - Uses heuristic filters (e.g., monetary and time constraints) to refine candidate results.

4. **API Layer (FastAPI):**  
   - Exposes a REST API for external access.
   - Automatically generates OpenAPI (Swagger) documentation for all endpoints.

## Getting Started

### Prerequisites

- Python 3.8 or later
- [PyTorch](https://pytorch.org/) (tested with version 1.8+)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)
- [NumPy](https://numpy.org/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/associative-neural-memory-db.git
   cd associative-neural-memory-db
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```bash
   pip install fastapi uvicorn pydantic numpy torch
   ```

## Running the Application

To start the server, run:

```bash
python ai_db_engine.py
```

The application will start on [http://localhost:8000](http://localhost:8000).

### API Documentation

Access the automatically generated API docs at:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## API Endpoints

### Create a New Record

- **URL:** `POST /records`
- **Body:**

  ```json
  {
    "data": "Order placed: $150",
    "metadata": {
      "timestamp": "2025-02-10T12:34:56Z",
      "amount": 150.0
    }
  }
  ```

- **Response:**

  ```json
  {
    "id": 1,
    "data": "Order placed: $150",
    "metadata": {
      "timestamp": "2025-02-10T12:34:56Z",
      "amount": 150.0
    },
    "similarity": null
  }
  ```

### Bulk Insert Records

- **URL:** `POST /records/bulk`
- **Body:** An array of record objects (same structure as single record creation).
- **Response:** A list of assigned record IDs.

### Retrieve a Record by ID

- **URL:** `GET /records/{record_id}`
- **Response:** Returns the record data, metadata, and an empty similarity field.

### Update a Record

- **URL:** `PUT /records/{record_id}`
- **Body:** Partial update object with optional `data` and/or `metadata`.
- **Response:** The updated record.

### Delete a Record

- **URL:** `DELETE /records/{record_id}`
- **Response:** A message confirming deletion.

### Query Records Using Natural Language

- **URL:** `POST /query`
- **Body:**

  ```json
  {
    "query": "Show me all orders above $100 placed last week.",
    "similarity_threshold": 0.5
  }
  ```

- **Response:** Returns a list of matching records with their similarity scores.

## Extending and Improving

- **Neural Embedding Model:**  
  Replace the simple character‑level model with a more advanced, pre‑trained model (e.g., using Hugging Face Transformers) for better embeddings.

- **Persistent Storage:**  
  Integrate a database (SQL, NoSQL, or a vector store) to persist data instead of using in‑memory storage.

- **Advanced NLP:**  
  Enhance the filter extraction logic with more sophisticated natural language processing.

- **Scalability:**  
  Adapt the engine for distributed and scalable environments, possibly using microservices.


## Acknowledgments

Inspired by human memory and the emerging need for AI‑driven, associative data storage systems. Special thanks to the developers and researchers working on neural networks, FastAPI, and open‑source AI tools.

