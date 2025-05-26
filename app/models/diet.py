from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from beanie import Document, Indexed

class DietCategory(BaseModel):
    id: str
    category: str

class NutritionalInfo(BaseModel):
    calories: Optional[int] = None
    protein_in_grams: Optional[float] = None
    carbohydrates_in_grams: Optional[float] = None
    fat_in_grams: Optional[float] = None
    fiber_in_grams: Optional[float] = None
    sugar_in_grams: Optional[float] = None
    sodium_in_mg: Optional[float] = None

class Diet(Document):
    # Required fields
    recipe: str
    category: DietCategory
    
    # Timing information
    prep_time_in_minutes: Optional[int] = None
    cook_time_in_minutes: Optional[int] = None
    total_time_in_minutes: Optional[int] = None
    
    # Serving information
    serving: Optional[int] = None
    difficulty: Optional[str] = None
    
    # Ingredients and directions
    all_ingredients: Optional[str] = None
    all_directions: Optional[str] = None
    
    # Nutritional information
    nutrition: NutritionalInfo = NutritionalInfo()
    
    # Additional metadata
    tags: List[str] = []
    cuisine_type: Optional[str] = None
    meal_type: List[str] = []  # breakfast, lunch, dinner, snack
    dietary_restrictions: List[str] = []  # vegetarian, vegan, gluten-free, etc.
    
    # Ratings and interactions
    average_rating: float = 0.0
    total_ratings: int = 0
    view_count: int = 0
    save_count: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "diets"
        indexes = [
            [("category.category", 1)],
            [("tags", 1)],
            [("cuisine_type", 1)],
            [("meal_type", 1)],
            [("dietary_restrictions", 1)],
            [("average_rating", -1)],
            [("created_at", -1)],
            [("recipe", "text"), ("all_ingredients", "text"), ("all_directions", "text")]
        ]

# Pydantic models for API
class DietCreate(BaseModel):
    recipe: str
    category: DietCategory
    prep_time_in_minutes: Optional[int] = None
    cook_time_in_minutes: Optional[int] = None
    serving: Optional[int] = None
    difficulty: Optional[str] = None
    all_ingredients: Optional[str] = None
    all_directions: Optional[str] = None
    nutrition: Optional[NutritionalInfo] = None
    tags: List[str] = []
    cuisine_type: Optional[str] = None
    meal_type: List[str] = []
    dietary_restrictions: List[str] = []

class DietUpdate(BaseModel):
    recipe: Optional[str] = None
    prep_time_in_minutes: Optional[int] = None
    cook_time_in_minutes: Optional[int] = None
    serving: Optional[int] = None
    difficulty: Optional[str] = None
    all_ingredients: Optional[str] = None
    all_directions: Optional[str] = None
    nutrition: Optional[NutritionalInfo] = None
    tags: Optional[List[str]] = None
    cuisine_type: Optional[str] = None
    meal_type: Optional[List[str]] = None
    dietary_restrictions: Optional[List[str]] = None

class DietResponse(BaseModel):
    id: str
    recipe: str
    category: DietCategory
    prep_time_in_minutes: Optional[int] = None
    cook_time_in_minutes: Optional[int] = None
    total_time_in_minutes: Optional[int] = None
    serving: Optional[int] = None
    difficulty: Optional[str] = None
    all_ingredients: Optional[str] = None
    all_directions: Optional[str] = None
    nutrition: NutritionalInfo
    tags: List[str] = []
    cuisine_type: Optional[str] = None
    meal_type: List[str] = []
    dietary_restrictions: List[str] = []
    average_rating: float = 0.0
    total_ratings: int = 0
    view_count: int = 0
    save_count: int = 0
    created_at: datetime

class DietSearchParams(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    cuisine_type: Optional[str] = None
    meal_type: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = None
    max_prep_time: Optional[int] = None
    max_cook_time: Optional[int] = None
    difficulty: Optional[str] = None
    min_rating: Optional[float] = None
    tags: Optional[List[str]] = None
    limit: int = 20
    skip: int = 0
    sort_by: str = "created_at"
    sort_order: str = "desc"
