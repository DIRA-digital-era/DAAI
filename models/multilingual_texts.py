from sqlalchemy import Column, Text, TIMESTAMP, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base

class MultilingualText(Base):
    __tablename__ = "multilingual_texts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(Text, nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=True)
    locale = Column(Text, nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP, nullable=False)
