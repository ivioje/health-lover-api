import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MLRecommendationService:
    def __init__(self):
        self.recipes_df = None
        self.content_model = None
        self.collaborative_model = None
        self.tfidf_vectorizer = None
        self.content_features = None
        self.user_item_matrix = None
        self.recipe_features = None
        self._load_data()
        self._build_models()
    
    def _load_data(self):
        """Load and preprocess the recipe data"""
        try:
            # Load the JSON data (in production, this could be from a database)
            data_path = Path(__file__).parent.parent.parent.parent / "api-diets.json"
            
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    recipes_data = json.load(f)
                
                # Convert to DataFrame
                self.recipes_df = pd.DataFrame(recipes_data)
                self._preprocess_data()
                logger.info(f"Loaded {len(self.recipes_df)} recipes")
            else:
                # Create sample data if file not found
                self._create_sample_data()
                logger.warning("Recipe data file not found, using sample data")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._create_sample_data()
    
    def _preprocess_data(self):
        """Preprocess the recipe data for ML models"""
        # Handle missing values
        self.recipes_df = self.recipes_df.fillna({
            'calories': 0,
            'protein_in_grams': 0,
            'carbohydrates_in_grams': 0,
            'fat_in_grams': 0,
            'difficulty': 'Easy',
            'prep_time_in_minutes': 30,
            'cook_time_in_minutes': 30,
            'serving': 1
        })
        
        # Create combined text features for content-based filtering
        text_features = []
        for _, row in self.recipes_df.iterrows():
            # Combine recipe name, ingredients, and category
            ingredients = []
            for i in range(1, 11):
                ingredient = row.get(f'ingredient_{i}', '')
                if ingredient and str(ingredient) != 'nan':
                    ingredients.append(str(ingredient))
            
            category_name = ''
            if isinstance(row.get('category'), dict):
                category_name = row['category'].get('category', '')
            
            text_feature = f"{row['recipe']} {' '.join(ingredients)} {category_name} {row['difficulty']}"
            text_features.append(text_feature)
        
        self.recipes_df['text_features'] = text_features
        
        # Create numerical features matrix
        numerical_features = ['calories', 'protein_in_grams', 'carbohydrates_in_grams', 
                             'fat_in_grams', 'prep_time_in_minutes', 'cook_time_in_minutes', 'serving']
        
        self.recipe_features = self.recipes_df[numerical_features].values
        
        # Normalize numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.recipe_features = scaler.fit_transform(self.recipe_features)
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        sample_data = [
            {
                "id": 1,
                "recipe": "Mediterranean Quinoa Bowl",
                "category": {"category": "Healthy Bowls"},
                "calories": 450,
                "protein_in_grams": 15,
                "carbohydrates_in_grams": 65,
                "fat_in_grams": 12,
                "difficulty": "Easy",
                "prep_time_in_minutes": 20,
                "cook_time_in_minutes": 15,
                "serving": 2,
                "ingredient_1": "quinoa",
                "ingredient_2": "olive oil",
                "ingredient_3": "vegetables"
            },
            {
                "id": 2,
                "recipe": "Keto Avocado Salad",
                "category": {"category": "Keto Recipes"},
                "calories": 350,
                "protein_in_grams": 8,
                "carbohydrates_in_grams": 12,
                "fat_in_grams": 30,
                "difficulty": "Easy",
                "prep_time_in_minutes": 10,
                "cook_time_in_minutes": 0,
                "serving": 1,
                "ingredient_1": "avocado",
                "ingredient_2": "cheese",
                "ingredient_3": "greens"
            }
        ]
        
        self.recipes_df = pd.DataFrame(sample_data)
        self._preprocess_data()
    
    def _build_models(self):
        """Build content-based and collaborative filtering models"""
        try:
            # Content-based model using TF-IDF
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.content_features = self.tfidf_vectorizer.fit_transform(self.recipes_df['text_features'])
            
            # Create sample user-item matrix for collaborative filtering
            self._create_user_item_matrix()
            
            # Build collaborative filtering model
            if self.user_item_matrix.shape[1] > 10:  # Only if we have enough recipes
                self.collaborative_model = TruncatedSVD(n_components=min(50, self.user_item_matrix.shape[1] - 1))
                self.collaborative_model.fit(self.user_item_matrix)
            
            logger.info("ML models built successfully")
            
        except Exception as e:
            logger.error(f"Error building models: {e}")
    
    def _create_user_item_matrix(self):
        """Create a sample user-item interaction matrix"""
        n_users = 1000
        n_recipes = len(self.recipes_df)
        
        # Create sparse matrix with random interactions
        np.random.seed(42)
        interactions = []
        
        for user_id in range(n_users):
            # Each user interacts with 5-15 recipes on average
            n_interactions = np.random.poisson(8)
            recipe_ids = np.random.choice(n_recipes, min(n_interactions, n_recipes), replace=False)
            
            for recipe_id in recipe_ids:
                # Rating between 1-5
                rating = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
                interactions.append([user_id, recipe_id, rating])
        
        interactions_df = pd.DataFrame(interactions, columns=['user_id', 'recipe_id', 'rating'])
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='recipe_id', 
            values='rating', 
            fill_value=0
        ).values
        
        self.user_item_matrix = csr_matrix(self.user_item_matrix)
    
    def get_content_based_recommendations(self, user_preferences: Dict[str, Any], n_recommendations: int = 10) -> List[Dict]:
        """Get content-based recommendations"""
        try:
            if self.content_features is None:
                return self._get_fallback_recommendations(n_recommendations)
            
            # Create user preference vector
            user_text = self._create_user_preference_text(user_preferences)
            user_vector = self.tfidf_vectorizer.transform([user_text])
            
            # Calculate similarities
            similarities = cosine_similarity(user_vector, self.content_features)[0]
            
            # Apply preference filters
            filtered_indices = self._apply_preference_filters(user_preferences)
            
            # Get top recommendations from filtered results
            filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
            filtered_similarities.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for i, score in filtered_similarities[:n_recommendations]:
                recipe = self.recipes_df.iloc[i].to_dict()
                recipe['recommendation_score'] = float(score)
                recipe['recommendation_type'] = 'content_based'
                recommendations.append(recipe)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def get_collaborative_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        try:
            if self.collaborative_model is None:
                return self._get_fallback_recommendations(n_recommendations)
            
            # For new users, use popularity-based recommendations
            user_id_int = hash(user_id) % self.user_item_matrix.shape[0]
            
            # Get user vector
            user_vector = self.user_item_matrix[user_id_int].toarray().flatten()
            
            # Transform to latent space
            user_latent = self.collaborative_model.transform([user_vector])
            
            # Get recommendations
            recipe_scores = self.collaborative_model.inverse_transform(user_latent)[0]
            
            # Get top unrated items
            unrated_indices = np.where(user_vector == 0)[0]
            unrated_scores = [(i, recipe_scores[i]) for i in unrated_indices if i < len(self.recipes_df)]
            unrated_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for i, score in unrated_scores[:n_recommendations]:
                recipe = self.recipes_df.iloc[i].to_dict()
                recipe['recommendation_score'] = float(score)
                recipe['recommendation_type'] = 'collaborative'
                recommendations.append(recipe)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def get_hybrid_recommendations(self, user_id: str, user_preferences: Dict[str, Any], n_recommendations: int = 10) -> List[Dict]:
        """Get hybrid recommendations combining content-based and collaborative filtering"""
        try:
            # Get both types of recommendations
            content_recs = self.get_content_based_recommendations(user_preferences, n_recommendations * 2)
            collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
            
            # Combine and weight the recommendations
            recipe_scores = {}
            
            # Weight content-based recommendations (60% weight)
            for rec in content_recs:
                recipe_id = rec['id']
                recipe_scores[recipe_id] = {
                    'recipe': rec,
                    'content_score': rec['recommendation_score'] * 0.6,
                    'collab_score': 0
                }
            
            # Add collaborative recommendations (40% weight)
            for rec in collab_recs:
                recipe_id = rec['id']
                if recipe_id in recipe_scores:
                    recipe_scores[recipe_id]['collab_score'] = rec['recommendation_score'] * 0.4
                else:
                    recipe_scores[recipe_id] = {
                        'recipe': rec,
                        'content_score': 0,
                        'collab_score': rec['recommendation_score'] * 0.4
                    }
            
            # Calculate final scores and sort
            final_recommendations = []
            for recipe_id, scores in recipe_scores.items():
                final_score = scores['content_score'] + scores['collab_score']
                recipe = scores['recipe'].copy()
                recipe['recommendation_score'] = final_score
                recipe['recommendation_type'] = 'hybrid'
                final_recommendations.append(recipe)
            
            final_recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return final_recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def get_personalized_recommendations(self, user_id: str, preferences: Dict[str, Any], 
                                       liked_recipes: List[int], num_recommendations: int = 10) -> List[Dict]:
        """Get personalized recommendations based on user preferences and liked recipes"""
        try:
            # Start with hybrid recommendations
            recommendations = self.get_hybrid_recommendations(user_id, preferences, num_recommendations * 2)
            
            # If user has liked recipes, use them to boost similar recipes
            if liked_recipes:
                similar_recipes = []
                for recipe_id in liked_recipes:
                    similar = self.get_similar_recipes(recipe_id, 5)
                    similar_recipes.extend(similar)
                
                # Boost scores for similar recipes
                similar_recipe_ids = {rec['id'] for rec in similar_recipes}
                for rec in recommendations:
                    if rec['id'] in similar_recipe_ids:
                        rec['recommendation_score'] *= 1.2
                        rec['reason'] = f"Similar to your liked recipes"
                    else:
                        rec['reason'] = f"Matches your {', '.join(preferences.get('health_goals', ['health']))} goals"
            else:
                # Add reasons for recommendations
                for rec in recommendations:
                    rec['reason'] = f"Matches your {', '.join(preferences.get('health_goals', ['health']))} goals"
            
            # Sort by updated scores
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            # Convert to expected format
            formatted_recs = []
            for rec in recommendations[:num_recommendations]:
                formatted_rec = {
                    "id": rec['id'],
                    "recipe": rec['recipe'],
                    "category": rec.get('category', {}).get('category') if isinstance(rec.get('category'), dict) else rec.get('category'),
                    "difficulty": rec.get('difficulty'),
                    "prep_time_in_minutes": rec.get('prep_time_in_minutes'),
                    "cook_time_in_minutes": rec.get('cook_time_in_minutes'),
                    "serving": rec.get('serving'),
                    "nutritional_info": {
                        "calories": rec.get('calories'),
                        "protein_in_grams": rec.get('protein_in_grams'),
                        "carbohydrates_in_grams": rec.get('carbohydrates_in_grams'),
                        "fat_in_grams": rec.get('fat_in_grams')
                    },
                    "confidence_score": min(rec['recommendation_score'], 1.0),
                    "reason": rec.get('reason', 'Personalized recommendation')
                }
                formatted_recs.append(formatted_rec)
            
            return formatted_recs
            
        except Exception as e:
            logger.error(f"Error in personalized recommendations: {e}")
            return self._get_fallback_formatted_recommendations(num_recommendations)
    
    def get_similar_recipes(self, recipe_id: int, num_recommendations: int = 5) -> List[Dict]:
        """Get recipes similar to a specific recipe"""
        try:
            if self.content_features is None:
                return self._get_fallback_formatted_recommendations(num_recommendations)
            
            # Find the recipe index
            recipe_indices = self.recipes_df[self.recipes_df['id'] == recipe_id].index
            if len(recipe_indices) == 0:
                return self._get_fallback_formatted_recommendations(num_recommendations)
            
            recipe_idx = recipe_indices[0]
            
            # Calculate similarities
            recipe_vector = self.content_features[recipe_idx]
            similarities = cosine_similarity([recipe_vector], self.content_features)[0]
            
            # Get most similar recipes (excluding the recipe itself)
            similar_indices = similarities.argsort()[::-1][1:num_recommendations + 1]
            
            recommendations = []
            for idx in similar_indices:
                recipe = self.recipes_df.iloc[idx].to_dict()
                formatted_rec = {
                    "id": recipe['id'],
                    "recipe": recipe['recipe'],
                    "category": recipe.get('category', {}).get('category') if isinstance(recipe.get('category'), dict) else recipe.get('category'),
                    "difficulty": recipe.get('difficulty'),
                    "prep_time_in_minutes": recipe.get('prep_time_in_minutes'),
                    "cook_time_in_minutes": recipe.get('cook_time_in_minutes'),
                    "serving": recipe.get('serving'),
                    "nutritional_info": {
                        "calories": recipe.get('calories'),
                        "protein_in_grams": recipe.get('protein_in_grams'),
                        "carbohydrates_in_grams": recipe.get('carbohydrates_in_grams'),
                        "fat_in_grams": recipe.get('fat_in_grams')
                    },
                    "confidence_score": float(similarities[idx]),
                    "reason": f"Similar ingredients and cooking method"
                }
                recommendations.append(formatted_rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting similar recipes: {e}")
            return self._get_fallback_formatted_recommendations(num_recommendations)
    
    def get_trending_recipes(self, num_recommendations: int = 10) -> List[Dict]:
        """Get trending/popular recipes based on nutritional balance"""
        try:
            # Calculate a trending score based on balanced nutrition
            self.recipes_df['trending_score'] = (
                # Balanced protein (20-30g gets highest score)
                np.where(self.recipes_df['protein_in_grams'].between(20, 30), 10, 
                        np.where(self.recipes_df['protein_in_grams'].between(15, 35), 7, 4)) +
                # Moderate calories (300-600 gets highest score)
                np.where(self.recipes_df['calories'].between(300, 600), 10,
                        np.where(self.recipes_df['calories'].between(200, 800), 7, 4)) +
                # Low carbs for keto (under 20g gets highest score)
                np.where(self.recipes_df['carbohydrates_in_grams'] < 20, 10,
                        np.where(self.recipes_df['carbohydrates_in_grams'] < 30, 5, 2)) +
                # Easy to make gets bonus
                np.where(self.recipes_df['difficulty'] == 'Easy', 5, 2)
            )
            
            trending_recipes = self.recipes_df.nlargest(num_recommendations, 'trending_score')
            
            recommendations = []
            for _, recipe in trending_recipes.iterrows():
                formatted_rec = {
                    "id": recipe['id'],
                    "recipe": recipe['recipe'],
                    "category": recipe.get('category', {}).get('category') if isinstance(recipe.get('category'), dict) else recipe.get('category'),
                    "difficulty": recipe.get('difficulty'),
                    "prep_time_in_minutes": recipe.get('prep_time_in_minutes'),
                    "cook_time_in_minutes": recipe.get('cook_time_in_minutes'),
                    "serving": recipe.get('serving'),
                    "nutritional_info": {
                        "calories": recipe.get('calories'),
                        "protein_in_grams": recipe.get('protein_in_grams'),
                        "carbohydrates_in_grams": recipe.get('carbohydrates_in_grams'),
                        "fat_in_grams": recipe.get('fat_in_grams')
                    },
                    "confidence_score": min(float(recipe['trending_score']) / 35, 1.0),
                    "reason": "Popular choice with balanced nutrition"
                }
                recommendations.append(formatted_rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting trending recipes: {e}")
            return self._get_fallback_formatted_recommendations(num_recommendations)
    
    def get_available_categories(self) -> List[str]:
        """Get list of available recipe categories"""
        try:
            if self.recipes_df is None:
                return ["Breakfast Recipes", "Lunch Recipes", "Dinner Recipes", "Snack Recipes"]
            
            categories = []
            for _, row in self.recipes_df.iterrows():
                category = row.get('category')
                if isinstance(category, dict) and 'category' in category:
                    cat_name = category['category']
                elif isinstance(category, str):
                    cat_name = category
                else:
                    continue
                
                if cat_name and cat_name not in categories:
                    categories.append(cat_name)
            
            return sorted(categories)
            
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return ["Breakfast Recipes", "Lunch Recipes", "Dinner Recipes", "Snack Recipes"]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the recommendation system"""
        try:
            stats = {
                "total_recipes": len(self.recipes_df) if self.recipes_df is not None else 0,
                "categories": len(self.get_available_categories()),
                "models_loaded": {
                    "content_model": self.content_features is not None,
                    "collaborative_model": self.collaborative_model is not None,
                    "tfidf_vectorizer": self.tfidf_vectorizer is not None
                },
                "data_summary": {}
            }
            
            if self.recipes_df is not None:
                stats["data_summary"] = {
                    "avg_calories": float(self.recipes_df['calories'].mean()),
                    "avg_protein": float(self.recipes_df['protein_in_grams'].mean()),
                    "avg_carbs": float(self.recipes_df['carbohydrates_in_grams'].mean()),
                    "avg_fat": float(self.recipes_df['fat_in_grams'].mean()),
                    "difficulty_distribution": self.recipes_df['difficulty'].value_counts().to_dict()
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": "Unable to retrieve stats", "total_recipes": 0}
    
    def _format_recommendations_for_api(self, recommendations: List[Dict]) -> List[Dict]:
        """Format recommendations for API response"""
        formatted = []
        
        for rec in recommendations:
            formatted_rec = {
                "id": rec['id'],
                "recipe": rec['recipe'],
                "category": rec.get('category', {}).get('category') if isinstance(rec.get('category'), dict) else rec.get('category'),
                "difficulty": rec.get('difficulty'),
                "prep_time_in_minutes": rec.get('prep_time_in_minutes'),
                "cook_time_in_minutes": rec.get('cook_time_in_minutes'),
                "serving": rec.get('serving'),
                "nutritional_info": {
                    "calories": rec.get('calories'),
                    "protein_in_grams": rec.get('protein_in_grams'),
                    "carbohydrates_in_grams": rec.get('carbohydrates_in_grams'),
                    "fat_in_grams": rec.get('fat_in_grams')
                },
                "confidence_score": min(rec.get('recommendation_score', 0.5), 1.0),
                "reason": rec.get('reason', f"Recommended based on {rec.get('recommendation_type', 'algorithm')}")
            }
            formatted.append(formatted_rec)
        
        return formatted
    
    def _get_fallback_formatted_recommendations(self, num_recommendations: int) -> List[Dict]:
        """Get fallback recommendations in the expected API format"""
        try:
            fallback_recs = self._get_fallback_recommendations(num_recommendations)
            return self._format_recommendations_for_api(fallback_recs)
        except Exception as e:
            logger.error(f"Error in fallback formatted recommendations: {e}")
            return []

    # Update existing methods to return formatted recommendations
    def get_content_based_recommendations(self, preferences: Dict[str, Any], num_recommendations: int = 10) -> List[Dict]:
        """Get content-based recommendations"""
        try:
            if self.content_features is None:
                return self._get_fallback_formatted_recommendations(num_recommendations)
            
            # Create user preference vector
            user_text = self._create_user_preference_text(preferences)
            user_vector = self.tfidf_vectorizer.transform([user_text])
            
            # Calculate similarities
            similarities = cosine_similarity(user_vector, self.content_features)[0]
            
            # Apply preference filters
            filtered_indices = self._apply_preference_filters(preferences)
            
            # Get top recommendations from filtered results
            filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
            filtered_similarities.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for i, score in filtered_similarities[:num_recommendations]:
                recipe = self.recipes_df.iloc[i].to_dict()
                recipe['recommendation_score'] = float(score)
                recipe['recommendation_type'] = 'content_based'
                recipe['reason'] = f"Matches your {', '.join(preferences.get('health_goals', ['health']))} preferences"
                recommendations.append(recipe)
            
            return self._format_recommendations_for_api(recommendations)
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return self._get_fallback_formatted_recommendations(num_recommendations)
    
    def get_collaborative_recommendations(self, user_id: str, num_recommendations: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        try:
            if self.collaborative_model is None:
                return self._get_fallback_formatted_recommendations(num_recommendations)
            
            # For new users, use popularity-based recommendations
            user_id_int = hash(user_id) % self.user_item_matrix.shape[0]
            
            # Get user vector
            user_vector = self.user_item_matrix[user_id_int].toarray().flatten()
            
            # Transform to latent space
            user_latent = self.collaborative_model.transform([user_vector])
            
            # Get recommendations
            recipe_scores = self.collaborative_model.inverse_transform(user_latent)[0]
            
            # Get top unrated items
            unrated_indices = np.where(user_vector == 0)[0]
            unrated_scores = [(i, recipe_scores[i]) for i in unrated_indices if i < len(self.recipes_df)]
            unrated_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for i, score in unrated_scores[:num_recommendations]:
                recipe = self.recipes_df.iloc[i].to_dict()
                recipe['recommendation_score'] = float(score)
                recipe['recommendation_type'] = 'collaborative'
                recipe['reason'] = 'Recommended based on similar users\' preferences'
                recommendations.append(recipe)
            
            return self._format_recommendations_for_api(recommendations)
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return self._get_fallback_formatted_recommendations(num_recommendations)
    
    def get_hybrid_recommendations(self, user_id: str, preferences: Dict[str, Any], 
                                 num_recommendations: int = 10, content_weight: float = 0.7) -> List[Dict]:
        """Get hybrid recommendations combining content-based and collaborative filtering"""
        try:
            # Get both types of recommendations
            content_recs = self.get_content_based_recommendations(preferences, num_recommendations * 2)
            collab_recs = self.get_collaborative_recommendations(user_id, num_recommendations * 2)
            
            # Combine and weight the recommendations
            recipe_scores = {}
            
            # Weight content-based recommendations
            for rec in content_recs:
                recipe_id = rec['id']
                recipe_scores[recipe_id] = {
                    'recipe_data': rec,
                    'content_score': rec['confidence_score'] * content_weight,
                    'collab_score': 0
                }
            
            # Add collaborative recommendations
            collab_weight = 1.0 - content_weight
            for rec in collab_recs:
                recipe_id = rec['id']
                if recipe_id in recipe_scores:
                    recipe_scores[recipe_id]['collab_score'] = rec['confidence_score'] * collab_weight
                else:
                    recipe_scores[recipe_id] = {
                        'recipe_data': rec,
                        'content_score': 0,
                        'collab_score': rec['confidence_score'] * collab_weight
                    }
            
            # Calculate final scores and sort
            final_recommendations = []
            for recipe_id, scores in recipe_scores.items():
                final_score = scores['content_score'] + scores['collab_score']
                recipe_data = scores['recipe_data'].copy()
                recipe_data['confidence_score'] = min(final_score, 1.0)
                recipe_data['reason'] = 'Hybrid recommendation combining your preferences and similar users'
                final_recommendations.append(recipe_data)
            
            final_recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            return final_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self._get_fallback_formatted_recommendations(num_recommendations)