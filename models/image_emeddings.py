#file: DAAI/models/image_embeddings.py
from sqlalchemy import Column, Text, Integer, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base

class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"
    image_id = Column(UUID(as_uuid=True), primary_key=True)
    embedding = Column("embedding", Text)  # We'll store vector as list in JSON for simplicity
    dimension = Column(Integer, default=1536)
    model_name = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False)
