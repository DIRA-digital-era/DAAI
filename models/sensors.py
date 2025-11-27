from sqlalchemy import Column, Text, TIMESTAMP, Numeric, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base

class Sensor(Base):
    __tablename__ = "sensors"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text)
    sensor_type = Column(Text)
    vendor = Column(Text)
    metadata = Column(JSON)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP, nullable=False)


class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sensor_id = Column(UUID(as_uuid=True), ForeignKey("sensors.id", ondelete="CASCADE"))
    reading_time = Column(TIMESTAMP, nullable=False)
    value = Column(Numeric)
    metadata = Column(JSON)
    linked_image_id = Column(UUID(as_uuid=True), ForeignKey("crop_images.id", ondelete="SET NULL"))
    created_at = Column(TIMESTAMP, nullable=False)
