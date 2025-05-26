import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, issparse, csr_matrix
import re
import implicit
import warnings
warnings.filterwarnings('ignore')

# Load file
json_file_path = 'api-diets.json'

# --- 1. Load Data ---
def load_data(file_path):
    """Load and validate JSON data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)

        if isinstance(recipes_data, list) and len(recipes_data) > 0:
            df = pd.DataFrame(recipes_data)
            print("DataFrame loaded successfully!")
            print(f"Initial DataFrame shape: {df.shape}")
            print("\nFirst 5 rows of the DataFrame:")
            print(df.head())
            return df
        else:
            print(f"Error: Expected JSON to be a non-empty list, but found {type(recipes_data)}.")
            return pd.DataFrame()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{file_path}': {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()

# Load the data
df = load_data(json_file_path)

if df.empty:
    print("Cannot proceed without data. Please check your JSON file.")
    exit()

# --- 2. Data Preprocessing ---
def preprocess_data(df):
    """Comprehensive data preprocessing"""
    
    # Extract category name from nested dictionary
    if 'category' in df.columns:
        df['category_name'] = df['category'].apply(
            lambda x: x.get('category', 'Unknown') if isinstance(x, dict) else 'Unknown'
        )
        print("Category column processed.")
    
    # Combine ingredient columns
    ingredient_cols = [f'ingredient_{i}' for i in range(1, 11) if f'ingredient_{i}' in df.columns]
    if ingredient_cols:
        df['all_ingredients'] = df[ingredient_cols].fillna('').astype(str).agg(' '.join, axis=1)
        df['all_ingredients'] = df['all_ingredients'].apply(
            lambda x: ' '.join(word for word in x.split() if word.lower() not in ['none', 'nan', ''])
        ).str.strip()
        print("Ingredient columns combined.")
    
    # Combine directions columns
    directions_cols = [f'directions_step_{i}' for i in range(1, 11) if f'directions_step_{i}' in df.columns]
    if directions_cols:
        df['all_directions'] = df[directions_cols].fillna('').astype(str).agg(' '.join, axis=1)
        df['all_directions'] = df['all_directions'].apply(
            lambda x: ' '.join(word for word in x.split() if word.lower() not in ['none', 'nan', ''])
        ).str.strip()
        print("Directions columns combined.")
    
    # Handle missing numerical values
    numerical_cols = ['prep_time_in_minutes', 'cook_time_in_minutes', 'serving', 
                     'fat_in_grams', 'calories', 'carbohydrates_in_grams', 'protein_in_grams']
    
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in '{col}' with median: {median_val}")
    
    # Handle missing categorical values
    categorical_cols = ['difficulty', 'category_name']
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in '{col}' with mode: {mode_val}")
    
    # Drop irrelevant columns
    columns_to_drop = [
        'prep_time_note', 'cook_time_note', 'image_attribution_name',
        'image_attribution_url', 'chef', 'source_url', 'category',
        'image_creative_commons', 'image'
    ]
    
    # Add ingredient and direction columns to drop list
    columns_to_drop.extend(ingredient_cols)
    columns_to_drop.extend(directions_cols)
    
    # Add measurement columns
    measurement_cols = [f'measurement_{i}' for i in range(1, 11) if f'measurement_{i}' in df.columns]
    columns_to_drop.extend(measurement_cols)
    
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    
    print(f"Dropped {len(columns_to_drop)} irrelevant columns.")
    return df

# Preprocess the data
df = preprocess_data(df)

# Verify required columns exist
required_columns = ['id', 'recipe']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    exit()

print("\nDataFrame after preprocessing:")
print(df.head(3))
print(f"Shape: {df.shape}")

# --- 3. Content-Based Filtering ---
def build_content_based_model(df):
    """Build content-based recommendation model"""
    
    # Define feature categories
    text_features = ['recipe', 'all_ingredients', 'all_directions']
    categorical_features = ['category_name', 'difficulty']
    numerical_features = [
        'prep_time_in_minutes', 'cook_time_in_minutes', 'serving',
        'calories', 'fat_in_grams', 'carbohydrates_in_grams', 'protein_in_grams'
    ]
    
    # Filter features that actually exist in the dataframe
    text_features = [col for col in text_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    numerical_features = [col for col in numerical_features if col in df.columns]
    
    # Ensure text columns are strings
    for col in text_features:
        df[col] = df[col].astype(str).fillna('')
    
    # Build preprocessor
    transformers = []
    
    if text_features:
        for col in text_features:
            max_features = 1000 if col == 'recipe' else 2000
            transformers.append((f'{col}_text', TfidfVectorizer(stop_words='english', max_features=max_features), col))
    
    if categorical_features:
        transformers.append(('cat_enc', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    
    if numerical_features:
        transformers.append(('num_scale', MinMaxScaler(), numerical_features))
    
    if not transformers:
        print("Error: No valid features found for preprocessing")
        return None, None
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    print("Building feature matrix...")
    try:
        feature_matrix = preprocessor.fit_transform(df)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Calculate cosine similarity
        print("Calculating cosine similarity...")
        cosine_sim = cosine_similarity(feature_matrix)
        print(f"Similarity matrix shape: {cosine_sim.shape}")
        
        return cosine_sim, preprocessor
        
    except Exception as e:
        print(f"Error building content-based model: {e}")
        return None, None

# Build content-based model
cosine_sim, preprocessor = build_content_based_model(df)

if cosine_sim is not None:
    # Create recipe ID to index mapping
    recipe_id_to_idx = pd.Series(df.index, index=df['id']).to_dict()
    
    def get_content_based_recommendations(recipe_id, num_recommendations=5):
        """Get content-based recommendations"""
        if recipe_id not in recipe_id_to_idx:
            print(f"Recipe ID {recipe_id} not found in the dataset.")
            return pd.DataFrame()
        
        idx = recipe_id_to_idx[recipe_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        
        recipe_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommended_recipes = df.iloc[recipe_indices].copy()
        recommended_recipes['similarity_score'] = similarity_scores
        
        return recommended_recipes[['id', 'recipe', 'category_name', 'similarity_score']]
    
    # Test content-based recommendations
    test_recipe_id = df['id'].iloc[0]
    print(f"\n--- Content-Based Recommendations for Recipe ID: {test_recipe_id} ---")
    content_recommendations = get_content_based_recommendations(test_recipe_id, 5)
    print(content_recommendations)

# --- 4. Collaborative Filtering ---
def generate_synthetic_interactions(df, num_users=100):
    """Generate synthetic user interactions for demonstration"""
    np.random.seed(42)
    
    user_interactions = []
    recipe_ids = df['id'].unique()
    
    for user_id in range(1, num_users + 1):
        num_interactions = np.random.randint(5, 21)
        user_recipes = np.random.choice(recipe_ids, size=min(num_interactions, len(recipe_ids)), replace=False)
        
        for recipe_id in user_recipes:
            user_interactions.append({
                'user_id': user_id,
                'recipe_id': int(recipe_id),
                'interaction': 1
            })
    
    return pd.DataFrame(user_interactions)

def build_collaborative_model(df, interactions_df):
    """Build collaborative filtering model"""
    
    # Create mappings
    unique_users = sorted(interactions_df['user_id'].unique())
    unique_recipes = sorted(interactions_df['recipe_id'].unique())  # Use only recipes in interactions_df
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    recipe_id_to_idx = {recipe_id: idx for idx, recipe_id in enumerate(unique_recipes)}
    idx_to_recipe_id = {idx: recipe_id for recipe_id, idx in recipe_id_to_idx.items()}
    
    # Create user-item matrix
    interactions_df['user_idx'] = interactions_df['user_id'].map(user_id_to_idx)
    interactions_df['recipe_idx'] = interactions_df['recipe_id'].map(recipe_id_to_idx)
    
    # Remove any rows where mapping failed
    interactions_df = interactions_df.dropna(subset=['user_idx', 'recipe_idx'])
    
    data = interactions_df['interaction'].values
    row_ind = interactions_df['user_idx'].values.astype(int)
    col_ind = interactions_df['recipe_idx'].values.astype(int)
    
    user_item_matrix = csr_matrix(
        (data, (row_ind, col_ind)), 
        shape=(len(unique_users), len(unique_recipes))
    )
    
    print(f"User-Item Matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4f}")
    
    # Train ALS model
    item_user_matrix = user_item_matrix.T.tocsr()
    
    model = implicit.als.AlternatingLeastSquares(
        factors=50,
        regularization=0.01,
        iterations=20,
        random_state=42
    )
    
    print("Training collaborative filtering model...")
    model.fit(item_user_matrix)
    print("Model trained successfully!")
    
    return model, user_item_matrix, user_id_to_idx, idx_to_recipe_id

# Generate synthetic interactions and build collaborative model
interactions_df = generate_synthetic_interactions(df, num_users=100)
print(f"Generated {len(interactions_df)} synthetic interactions")

try:
    collab_model, user_item_matrix, user_id_to_idx, idx_to_recipe_id = build_collaborative_model(df, interactions_df)
    
    def get_collaborative_recommendations(user_id, num_recommendations=5):
        """Get collaborative filtering recommendations"""
        if user_id not in user_id_to_idx:
            print(f"User ID {user_id} not found in the interaction data.")
            return pd.DataFrame()
        
        user_idx = user_id_to_idx[user_id]
        
        try:
            recommendations = collab_model.recommend(
                user_idx,
                user_item_matrix[user_idx],
                N=num_recommendations
            )
            
            recommended_recipe_indices = [r[0] for r in recommendations]
            scores = [r[1] for r in recommendations]
            
            # Map indices back to recipe IDs
            recommended_recipe_ids = [idx_to_recipe_id[idx] for idx in recommended_recipe_indices if idx in idx_to_recipe_id]
            
            recommended_recipes = df[df['id'].isin(recommended_recipe_ids)].copy()
            
            # Add scores in the correct order
            score_map = dict(zip(recommended_recipe_ids, scores))
            recommended_recipes['predicted_score'] = recommended_recipes['id'].map(score_map)
            
            return recommended_recipes[['id', 'recipe', 'category_name', 'predicted_score']].sort_values('predicted_score', ascending=False)
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return pd.DataFrame()
    
    # Test collaborative recommendations
    test_user_id = list(user_id_to_idx.keys())[0]
    print(f"\n--- Collaborative Recommendations for User ID: {test_user_id} ---")
    collab_recommendations = get_collaborative_recommendations(test_user_id, 5)
    print(collab_recommendations)
    
except Exception as e:
    print(f"Error building collaborative filtering model: {e}")