from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from pgvector.sqlalchemy import Vector
from src.config import DATABASE_URL, EMBEDDING_DIM
from datetime import datetime

class Base(DeclarativeBase):
    pass

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    height = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    training_years = Column(Float, nullable=False)
    goal = Column(String(20), nullable=False)
    available_equipment = Column(JSON, nullable=False)
    days_per_week = Column(Integer, nullable=False)
    injuries = Column(JSON, default=[])
    preferences = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Exercise(Base):
    __tablename__ = "exercises"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    name_en = Column(String(100))
    exercise_type = Column(String(20))
    difficulty = Column(String(10))
    equipment = Column(String(50))
    target_muscles = Column(JSON)
    description = Column(Text)
    common_errors = Column(JSON)
    embedding = Column(Vector(EMBEDDING_DIM))

class TrainingPlan(Base):
    __tablename__ = "training_plans"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    goal = Column(String(20))
    plan_data = Column(JSON, nullable=False)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)
