from sqlalchemy import Column, Text, TIMESTAMP, Integer, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base

class CropImage(Base):
    __tablename__ = "crop_images"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    crop_id = Column(UUID(as_uuid=True), ForeignKey("crops.id", ondelete="SET NULL"))
    variety_id = Column(UUID(as_uuid=True), ForeignKey("crop_varieties.id", ondelete="SET NULL"))
    harvest_id = Column(UUID(as_uuid=True), nullable=True)  # assuming harvest_logs not implemented yet
    storage_object_id = Column(UUID(as_uuid=True), nullable=True)
    file_name = Column(Text)
    capture_time = Column(TIMESTAMP)
    uploader_id = Column(UUID(as_uuid=True))
    metadata_json = Column(JSON)
    image_width = Column(Integer)
    image_height = Column(Integer)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP, nullable=False)
