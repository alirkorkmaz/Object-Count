from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import datetime

Base = declarative_base()

class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    video_name = Column(String(255), nullable=False)
    model_used = Column(String(50), nullable=False)
    tracker_used = Column(String(50), nullable=False)
    object_id = Column(Integer, nullable=False)
    object_label = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=func.now()) 
    current_total_count = Column(Integer, nullable=False, default=0)

    def __repr__(self):
        return f"<DetectionRecord(id={self.id}, object_id={self.object_id}, label='{self.object_label}', timestamp='{self.timestamp}')>"

class OverallCount(Base):
    __tablename__ = "overall_counts"

    id = Column(Integer, primary_key=True, index=True)
    video_name = Column(String(255), nullable=False)
    model_used = Column(String(50), nullable=False)
    tracker_used = Column(String(50), nullable=False)
    final_count = Column(Integer, nullable=False)
    start_time = Column(DateTime(timezone=True), default=func.now())
    end_time = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    line_coordinates = Column(Text, nullable=True) # JSON string olarak
    processed_video_path = Column(String(255), nullable=True) # İşlenmiş videonun yolu
    
    def __repr__(self):
        return f"<OverallCount(id={self.id}, video='{self.video_name}', final_count={self.final_count})>"