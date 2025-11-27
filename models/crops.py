from sqlalchemy import Column, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base

class Crop(Base):
    __tablename__ = "crops"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scientific_name = Column(Text)
    common_name = Column(Text)
    crop_type = Column(Text)
    description = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP, nullable=False)


class CropVariety(Base):
    __tablename__ = "crop_varieties"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    crop_id = Column(UUID(as_uuid=True), nullable=False)
    variety_name = Column(Text)
    description = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP, nullable=False)
