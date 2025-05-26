from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from beanie import Document, Indexed
from enum import Enum

class InteractionType(str, Enum):
    VIEW = "view"
    LIKE = "like"
    SAVE = "save"
    RATING = "rating"
    SHARE = "share"

class UserInteraction(Document):
    user_id: Indexed(str)
    diet_id: Indexed(str)
    interaction_type: InteractionType
    value: Optional[float] = None  # For ratings (1-5) or other numeric values
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    
    class Settings:
        name = "user_interactions"
        indexes = [
            [("user_id", 1), ("diet_id", 1), ("interaction_type", 1)],
            [("user_id", 1), ("timestamp", -1)],
            [("diet_id", 1), ("timestamp", -1)],
            [("timestamp", -1)],
        ]

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    recommendation_type: str = "hybrid"  # "content", "collaborative", "hybrid"
    filters: Optional[Dict[str, Any]] = None

class RecommendationResponse(BaseModel):
    diet_id: str
    recipe: str
    category: str
    score: float
    recommendation_type: str
    reasoning: Optional[str] = None

class RecommendationsResult(BaseModel):
    user_id: str
    recommendations: List[RecommendationResponse]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "1.0"
    request_params: Dict[str, Any]

class UserPreferencesProfile(BaseModel):
    """Derived user preferences from interactions"""
    user_id: str
    preferred_categories: List[str] = []
    preferred_cuisines: List[str] = []
    preferred_meal_types: List[str] = []
    dietary_restrictions: List[str] = []
    avg_rating_given: float = 0.0
    favorite_ingredients: List[str] = []
    interaction_count: int = 0
    last_active: datetime = Field(default_factory=datetime.utcnow)

# API Models for interactions
class InteractionCreate(BaseModel):
    diet_id: str
    interaction_type: InteractionType
    value: Optional[float] = None
    session_id: Optional[str] = None

class InteractionResponse(BaseModel):
    id: str
    user_id: str
    diet_id: str
    interaction_type: InteractionType
    value: Optional[float] = None
    timestamp: datetime

class UserStats(BaseModel):
    user_id: str
    total_views: int = 0
    total_likes: int = 0
    total_saves: int = 0
    total_ratings: int = 0
    avg_rating_given: float = 0.0
    unique_categories_explored: int = 0
    last_activity: Optional[datetime] = None

class DietStats(BaseModel):
    diet_id: str
    total_views: int = 0
    total_likes: int = 0
    total_saves: int = 0
    total_ratings: int = 0
    avg_rating: float = 0.0
    unique_users_interacted: int = 0
    last_interaction: Optional[datetime] = None
