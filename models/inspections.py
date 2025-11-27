from sqlalchemy import Column, Text, TIMESTAMP, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base

class Inspection(Base):
    __tablename__ = "inspections"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    crop_image_id = Column(UUID(as_uuid=True), ForeignKey("crop_images.id", ondelete="CASCADE"))
    inspector_type = Column(Text, default="ai")
    inspector_id = Column(UUID(as_uuid=True), nullable=True)
    detection_time = Column(TIMESTAMP, nullable=False)
    health_status = Column(Text)
    diagnosis = Column(JSON)
    remedy_text_id = Column(UUID(as_uuid=True), nullable=True)
    model_version = Column(Text)
    locale = Column(Text, default="en")
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP, nullable=False)


class InspectionReference(Base):
    __tablename__ = "inspection_references"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inspection_id = Column(UUID(as_uuid=True), ForeignKey("inspections.id", ondelete="CASCADE"))
    referenced_type = Column(Text, nullable=False)
    referenced_id = Column(UUID(as_uuid=True), nullable=False)
    rank = Column(Integer, nullable=False)
    score = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
