#DAAI/models/audit_meta.py
from sqlalchemy import Column, Text, TIMESTAMP, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base

class AuditMeta(Base):
    __tablename__ = "audit_meta"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    table_name = Column(Text, nullable=False)
    record_id = Column(UUID(as_uuid=True), nullable=False)
    action = Column(Text, nullable=False)
    payload = Column(JSON)
    created_at = Column(TIMESTAMP, nullable=False)
