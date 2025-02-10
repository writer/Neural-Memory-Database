import re
import string
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ------------------------------------------------------
# Neural Embedding Model (Character‑Level)
# ------------------------------------------------------
class NeuralEmbedder(nn.Module):
    def __init__(self, char_vocab_size: int, char_embed_dim: int, hidden_dim: int, output_dim: int):
        """
        A simple feed‑forward neural network that embeds text (represented as a sequence of character indices)
        into a fixed‑size vector.
        """
        super(NeuralEmbedder, self).__init__()
        # Embedding layer for characters (0 is reserved for padding)
        self.embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        # A small feed‑forward network
        self.fc1 = nn.Linear(char_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len]
        embeds = self.embedding(x)            # -> [batch, seq_len, char_embed_dim]
        pooled = embeds.mean(dim=1)           # mean pooling over the sequence -> [batch, char_embed_dim]
        hidden = F.relu(self.fc1(pooled))     # -> [batch, hidden_dim]
        output = self.fc2(hidden)             # -> [batch, output_dim]
        return output

# ------------------------------------------------------
# Heuristic Filter Extraction from Natural Language Queries
# ------------------------------------------------------
def extract_filters(nl_query: str) -> Dict[str, Any]:
    """
    Extract simple constraints from a natural language query.
    
    Supported heuristics:
      - Monetary constraint: e.g., "above $100" extracts an amount_threshold of 100.
      - Time constraint: if "last week" appears, set a time range to the past 7 days.
    
    Returns a dict with:
      - "amount_threshold": a float or None,
      - "time_range": a dict with keys "start" and "end" (as datetime objects) or None.
    """
    filters = {}
    
    # Monetary filter: look for "above $<number>"
    amount_match = re.search(r"above \$([\d\.]+)", nl_query, re.IGNORECASE)
    if amount_match:
        try:
            filters["amount_threshold"] = float(amount_match.group(1))
        except Exception:
            filters["amount_threshold"] = None
    else:
        filters["amount_threshold"] = None
    
    # Time filter: if "last week" is in the query, use the past 7 days.
    if "last week" in nl_query.lower():
        now = datetime.utcnow()
        filters["time_range"] = {
            "start": now - timedelta(days=7),
            "end": now
        }
    else:
        filters["time_range"] = None
        
    return filters

# ------------------------------------------------------
# Associative Neural Memory Engine with Integrated Neural Embedding
# ------------------------------------------------------
class AssociativeMemoryEngine:
    def __init__(self, vector_dim: int = 128, max_seq_len: int = 100):
        self.vector_dim = vector_dim
        self.records: Dict[int, Dict[str, Any]] = {}  # In‑memory storage: record_id -> record dict
        self.next_id = 1
        self.max_seq_len = max_seq_len

        # Build a character vocabulary from a fixed set of characters.
        # We include lowercase letters, digits, punctuation, and space.
        chars = string.ascii_lowercase + string.digits + string.punctuation + " "
        self.char_vocab = {ch: i+1 for i, ch in enumerate(chars)}  # index 0 reserved for padding
        self.vocab_size = len(self.char_vocab) + 1

        # Initialize the neural embedder.
        # For demonstration, we use a small network with random weights.
        self.embedder = NeuralEmbedder(
            char_vocab_size=self.vocab_size,
            char_embed_dim=16,
            hidden_dim=64,
            output_dim=self.vector_dim
        )
        # In production, you would load pre-trained weights.
        self.embedder.eval()  # Set model to evaluation mode

    def text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Convert input text into a tensor of character indices.
        The text is lowercased, tokenized at the character level, and padded/truncated to max_seq_len.
        """
        text = text.lower()
        indices = [self.char_vocab.get(ch, 0) for ch in text]
        if len(indices) < self.max_seq_len:
            indices = indices + [0] * (self.max_seq_len - len(indices))
        else:
            indices = indices[:self.max_seq_len]
        return torch.tensor([indices], dtype=torch.long)  # shape: [1, max_seq_len]

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute a neural embedding for the given text using the internal neural model.
        Returns a numpy array of shape (vector_dim,).
        """
        tensor = self.text_to_tensor(text)  # Convert text to tensor indices.
        with torch.no_grad():
            embedding_tensor = self.embedder(tensor)  # [1, vector_dim]
        return embedding_tensor.squeeze(0).cpu().numpy()

    def insert(self, data: str, metadata: Dict[str, Any]) -> int:
        """
        Insert a new record into the engine.
        Each record includes:
          - data: free‑form text,
          - metadata: e.g. {"timestamp": datetime, "amount": 150},
          - embedding: computed from the text via the neural embedder.
        Returns the assigned record ID.
        """
        embedding = self.compute_embedding(data)
        record = {
            "id": self.next_id,
            "data": data,
            "metadata": metadata,
            "embedding": embedding
        }
        self.records[self.next_id] = record
        self.next_id += 1
        return record["id"]

    def bulk_insert(self, records: List[Dict[str, Any]]) -> List[int]:
        """
        Bulk insert a list of records.
        Each record must be a dict with keys "data" and "metadata".
        Returns a list of assigned record IDs.
        """
        ids = []
        for rec in records:
            rec_id = self.insert(rec["data"], rec["metadata"])
            ids.append(rec_id)
        return ids

    def update(self, record_id: int, new_data: Optional[str] = None, new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing record.
          - new_data (if provided) updates the text and re-computes its embedding.
          - new_metadata (if provided) is merged with the existing metadata.
        Returns True if the update succeeds; otherwise False.
        """
        if record_id not in self.records:
            return False
        record = self.records[record_id]
        if new_data is not None:
            record["data"] = new_data
            record["embedding"] = self.compute_embedding(new_data)
        if new_metadata is not None:
            record["metadata"].update(new_metadata)
        return True

    def delete(self, record_id: int) -> bool:
        """
        Delete a record by its ID.
        Returns True if deletion was successful, False if the record does not exist.
        """
        if record_id in self.records:
            del self.records[record_id]
            return True
        return False

    def query(self, nl_query: str, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Process a natural language query using an associative (content‑addressable) approach:
          1. Convert the query to a neural embedding.
          2. Compute cosine similarity between the query embedding and each record.
          3. Extract heuristic filters (e.g., "above $100", "last week") from the query.
          4. Apply these filters to candidate records.
        Returns a list of matching records (with similarity scores).
        """
        query_embedding = self.compute_embedding(nl_query)
        candidates = []
        for record in self.records.values():
            score = cosine_similarity(query_embedding, record["embedding"])
            if score >= similarity_threshold:
                candidates.append((record, score))
        # Sort candidates by descending similarity.
        candidates.sort(key=lambda x: x[1], reverse=True)
        filters = extract_filters(nl_query)
        final_results = []
        for record, score in candidates:
            valid = True
            # Apply monetary filter if specified.
            if filters.get("amount_threshold") is not None:
                if "amount" in record["metadata"]:
                    if record["metadata"]["amount"] <= filters["amount_threshold"]:
                        valid = False
                else:
                    valid = False
            # Apply time filter if specified.
            if filters.get("time_range") is not None:
                if "timestamp" in record["metadata"]:
                    rec_time = record["metadata"]["timestamp"]
                    time_range = filters["time_range"]
                    if not (time_range["start"] <= rec_time <= time_range["end"]):
                        valid = False
                else:
                    valid = False
            if valid:
                final_results.append({
                    "id": record["id"],
                    "data": record["data"],
                    "metadata": record["metadata"],
                    "similarity": score
                })
        return final_results

# ------------------------------------------------------
# FastAPI Application and API Schema
# ------------------------------------------------------
app = FastAPI(
    title="Associative Neural Memory DB",
    description="A new‑generation AI DB engine inspired by human memory storage. It embeds text using an internal neural model and retrieves records using natural language queries.",
    version="1.0.0"
)

# Pydantic models for the API
class RecordMetadata(BaseModel):
    timestamp: datetime = Field(..., example="2025-02-10T12:34:56Z")
    amount: Optional[float] = Field(None, example=150.0)

class RecordCreate(BaseModel):
    data: str = Field(..., example="Order placed: $150")
    metadata: RecordMetadata

class RecordUpdate(BaseModel):
    data: Optional[str] = Field(None, example="Order placed: $200")
    metadata: Optional[RecordMetadata] = None

class RecordOut(BaseModel):
    id: int
    data: str
    metadata: RecordMetadata
    similarity: Optional[float] = None

class QueryRequest(BaseModel):
    query: str = Field(..., example="Show me all orders above $100 placed last week.")
    similarity_threshold: Optional[float] = Field(0.5, example=0.5)

class QueryResponse(BaseModel):
    results: List[RecordOut]

# Initialize our engine instance.
engine = AssociativeMemoryEngine(vector_dim=128, max_seq_len=100)

# API Endpoints

@app.post("/records", response_model=RecordOut, summary="Create a new record")
def create_record(record: RecordCreate):
    rec_id = engine.insert(record.data, record.metadata.dict())
    stored = engine.records[rec_id]
    return {
        "id": stored["id"],
        "data": stored["data"],
        "metadata": stored["metadata"],
        "similarity": None
    }

@app.post("/records/bulk", response_model=List[int], summary="Bulk insert records")
def bulk_insert(records: List[RecordCreate]):
    recs = [record.dict() for record in records]
    ids = engine.bulk_insert(recs)
    return ids

@app.get("/records/{record_id}", response_model=RecordOut, summary="Retrieve a record by ID")
def get_record(record_id: int):
    if record_id not in engine.records:
        raise HTTPException(status_code=404, detail="Record not found")
    rec = engine.records[record_id]
    return {
        "id": rec["id"],
        "data": rec["data"],
        "metadata": rec["metadata"],
        "similarity": None
    }

@app.put("/records/{record_id}", response_model=RecordOut, summary="Update a record by ID")
def update_record(record_id: int, record_update: RecordUpdate):
    success = engine.update(
        record_id,
        new_data=record_update.data,
        new_metadata=record_update.metadata.dict() if record_update.metadata else None
    )
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    rec = engine.records[record_id]
    return {
        "id": rec["id"],
        "data": rec["data"],
        "metadata": rec["metadata"],
        "similarity": None
    }

@app.delete("/records/{record_id}", summary="Delete a record by ID")
def delete_record(record_id: int):
    success = engine.delete(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"detail": f"Record {record_id} deleted successfully"}

@app.post("/query", response_model=QueryResponse, summary="Query records using natural language")
def query_records(query_request: QueryRequest):
    results = engine.query(query_request.query, similarity_threshold=query_request.similarity_threshold)
    return {"results": results}

# ------------------------------------------------------
# Run the Application
# ------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
