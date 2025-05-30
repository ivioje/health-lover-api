from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from ..services.ml_service import MLRecommendationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Initialize ML service
ml_service = MLRecommendationService()

# Request models
class UserPreferences(BaseModel):
    dietary_restrictions: List[str] = Field(default=[], description="List of dietary restrictions")
    health_goals: List[str] = Field(default=[], description="List of health goals")
    activity_level: str = Field(default="moderate", description="User's activity level")
    preferred_cuisines: List[str] = Field(default=[], description="Preferred cuisine types")
    disliked_ingredients: List[str] = Field(default=[], description="Disliked ingredients")

class ContentBasedRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    preferences: UserPreferences = Field(..., description="User preferences")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")

class CollaborativeRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")

class HybridRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    preferences: UserPreferences = Field(..., description="User preferences")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    content_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for content-based recommendations")

class PersonalizedRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    preferences: UserPreferences = Field(..., description="User preferences")
    liked_recipes: List[int] = Field(default=[], description="List of recipe IDs the user has liked")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")

# Response models
class NutritionalInfo(BaseModel):
    calories: Optional[float] = None
    protein_in_grams: Optional[float] = None
    carbohydrates_in_grams: Optional[float] = None
    fat_in_grams: Optional[float] = None

class RecipeRecommendation(BaseModel):
    id: int
    recipe: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    prep_time_in_minutes: Optional[int] = None
    cook_time_in_minutes: Optional[int] = None
    serving: Optional[int] = None
    nutritional_info: NutritionalInfo
    confidence_score: float = Field(..., description="Recommendation confidence score")
    reason: str = Field(..., description="Reason for recommendation")

class RecommendationResponse(BaseModel):
    recommendations: List[RecipeRecommendation]
    total_count: int
    algorithm_used: str
    user_id: str

@router.get("/health")
async def health_check():
    """Health check endpoint for the recommendation service"""
    return {
        "status": "healthy",
        "service": "recommendation",
        "model_loaded": ml_service.df is not None,
        "total_recipes": len(ml_service.df) if ml_service.df is not None else 0
    }

@router.post("/content-based", response_model=RecommendationResponse)
async def get_content_based_recommendations(request: ContentBasedRequest):
    """
    Get content-based recommendations based on user preferences
    """
    try:
        # Convert preferences to dictionary for ML service
        preferences_dict = {
            "dietary_restrictions": request.preferences.dietary_restrictions,
            "health_goals": request.preferences.health_goals,
            "activity_level": request.preferences.activity_level,
            "preferred_cuisines": request.preferences.preferred_cuisines,
            "disliked_ingredients": request.preferences.disliked_ingredients
        }
        
        recommendations = ml_service.get_content_based_recommendations(
            preferences=preferences_dict,
            num_recommendations=request.num_recommendations
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            algorithm_used="content-based",
            user_id=request.user_id
        )
        
    except Exception as e:
        logger.error(f"Error getting content-based recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post("/collaborative", response_model=RecommendationResponse)
async def get_collaborative_recommendations(request: CollaborativeRequest):
    """
    Get collaborative filtering recommendations
    """
    try:
        recommendations = ml_service.get_collaborative_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            algorithm_used="collaborative-filtering",
            user_id=request.user_id
        )
        
    except Exception as e:
        logger.error(f"Error getting collaborative recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post("/hybrid", response_model=RecommendationResponse)
async def get_hybrid_recommendations(request: HybridRequest):
    """
    Get hybrid recommendations combining content-based and collaborative filtering
    """
    try:
        preferences_dict = {
            "dietary_restrictions": request.preferences.dietary_restrictions,
            "health_goals": request.preferences.health_goals,
            "activity_level": request.preferences.activity_level,
            "preferred_cuisines": request.preferences.preferred_cuisines,
            "disliked_ingredients": request.preferences.disliked_ingredients
        }
        
        recommendations = ml_service.get_hybrid_recommendations(
            user_id=request.user_id,
            preferences=preferences_dict,
            num_recommendations=request.num_recommendations,
            content_weight=request.content_weight
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            algorithm_used="hybrid",
            user_id=request.user_id
        )
        
    except Exception as e:
        logger.error(f"Error getting hybrid recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post("/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(request: PersonalizedRequest):
    """
    Get personalized recommendations based on user preferences and interaction history
    """
    try:
        preferences_dict = {
            "dietary_restrictions": request.preferences.dietary_restrictions,
            "health_goals": request.preferences.health_goals,
            "activity_level": request.preferences.activity_level,
            "preferred_cuisines": request.preferences.preferred_cuisines,
            "disliked_ingredients": request.preferences.disliked_ingredients
        }
        
        recommendations = ml_service.get_personalized_recommendations(
            user_id=request.user_id,
            preferences=preferences_dict,
            liked_recipes=request.liked_recipes,
            num_recommendations=request.num_recommendations
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            algorithm_used="personalized",
            user_id=request.user_id
        )
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.get("/similar/{recipe_id}")
async def get_similar_recipes(recipe_id: int, num_recommendations: int = 5):
    """
    Get recipes similar to a specific recipe
    """
    try:
        recommendations = ml_service.get_similar_recipes(
            recipe_id=recipe_id,
            num_recommendations=num_recommendations
        )
        
        return {
            "recipe_id": recipe_id,
            "similar_recipes": recommendations,
            "total_count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Error getting similar recipes: {e}")
        raise HTTPException(status_code=500, detail=f"Error finding similar recipes: {str(e)}")

@router.get("/trending")
async def get_trending_recipes(num_recommendations: int = 10):
    """
    Get trending/popular recipes
    """
    try:
        recommendations = ml_service.get_trending_recipes(
            num_recommendations=num_recommendations
        )
        
        return {
            "trending_recipes": recommendations,
            "total_count": len(recommendations),
            "algorithm_used": "trending"
        }
        
    except Exception as e:
        logger.error(f"Error getting trending recipes: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting trending recipes: {str(e)}")

@router.get("/categories")
async def get_recipe_categories():
    """
    Get available recipe categories
    """
    try:
        categories = ml_service.get_available_categories()
        return {
            "categories": categories,
            "total_count": len(categories)
        }
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")

@router.get("/stats")
async def get_recommendation_stats():
    """
    Get statistics about the recommendation system
    """
    try:
        stats = ml_service.get_system_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")