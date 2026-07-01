from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UserProfileInput(BaseModel):
    height: float = Field(..., ge=100, le=250, description="身高(cm)")
    weight: float = Field(..., ge=30, le=200, description="体重(kg)")
    training_years: float = Field(..., ge=0, le=30, description="训练年限")
    goal: str = Field(..., pattern="^(增肌|减脂)$")
    available_equipment: List[str] = Field(..., min_length=1)
    days_per_week: int = Field(..., ge=1, le=7)
    injuries: List[str] = Field(default=[])
    preferences: dict = Field(default={})

class ExerciseItem(BaseModel):
    name: str
    sets: int = Field(..., ge=1, le=10)
    reps: str
    rest: str
    notes: str = ""

class TrainingDay(BaseModel):
    day: int
    focus: str
    exercises: List[ExerciseItem]

class TrainingPlanOutput(BaseModel):
    plan_id: str
    user_id: int
    goal: str
    weeks: int
    sessions_per_week: int
    days: List[TrainingDay]
    warnings: List[str] = []

class ExerciseAnalysisInput(BaseModel):
    exercise_name: str
    user_description: str
    user_level: str = "中级"

class ExerciseAnalysisOutput(BaseModel):
    exercise_name: str
    issues_found: List[str]
    severity: str
    suggestions: List[str]
    confidence: float

class PlanRequest(BaseModel):
    user_profile: UserProfileInput
    query: str = ""

class AnalysisRequest(BaseModel):
    analysis: ExerciseAnalysisInput

class SearchResult(BaseModel):
    content: str
    score: float
    source: str
    metadata: dict = {}
