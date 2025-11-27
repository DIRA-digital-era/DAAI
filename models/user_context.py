from sqlalchemy import Column, TIMESTAMP, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base

class UserContext(Base):
    __tablename__ = "user_context"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    last_embedding = Column("last_embedding", JSON)
    last_request_time = Column(TIMESTAMP, nullable=False)
    metadata = Column(JSON)
