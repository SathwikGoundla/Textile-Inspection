"""Database ORM Models"""
from sqlalchemy import Column, String, Float, DateTime, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class InspectionLog(Base):
    __tablename__ = "inspection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(String(36), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(20))        # PASS / FAIL / REJECTED
    textile_type = Column(String(100), nullable=True)
    confidence = Column(Float, default=0.0)
    defects = Column(Text, default="[]")   # JSON string
    severity = Column(String(20), default="NONE")
