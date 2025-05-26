# Health Lover Recommendation API

A pure Machine Learning recommendation service for the Health Lover application. This API focuses solely on providing diet and recipe recommendations using content-based filtering, collaborative filtering, and hybrid approaches.

## Features

- **Content-Based Recommendations**: Find similar recipes based on ingredients, cuisine, and nutritional content
- **Collaborative Filtering**: Recommend recipes based on user behavior patterns
- **Hybrid Recommendations**: Combine both approaches for better accuracy
- **User Interaction Tracking**: Record user interactions to improve recommendations
- **Model Statistics**: Get insights into recommendation model performance

## Architecture

This API is designed to work alongside a Next.js frontend that handles:
- User authentication and management
- Recipe/diet CRUD operations
- UI components and user interface

The API focuses purely on:
- ML model inference
- Recommendation generation
- User interaction tracking for model improvement

## Quick Start

### Prerequisites

- Python 3.8+
- MongoDB (for storing interaction data)

### Installation

1. **Clone and navigate to the API directory**:
```bash
cd health-lover-api
```

2. **Create virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
copy .env.example .env
# Edit .env with your configuration
```

5. **Run the API**:
```bash
python app/main.py
# or
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

## API Endpoints

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /api/v1/info` - Detailed API information

### Recommendation Endpoints

- `POST /api/v1/recommendations/content-based` - Get content-based recommendations
- `POST /api/v1/recommendations/collaborative` - Get collaborative filtering recommendations  
- `POST /api/v1/recommendations/hybrid` - Get hybrid recommendations
- `POST /api/v1/recommendations/similar-recipes/{recipe_id}` - Find similar recipes
- `GET /api/v1/recommendations/stats` - Get recommendation statistics

### Interaction Endpoints

- `POST /api/v1/recommendations/interactions` - Record user interaction
- `GET /api/v1/recommendations/user/{user_id}/interactions` - Get user interaction history

### Admin Endpoints

- `POST /api/v1/recommendations/retrain` - Trigger model retraining

## Usage Examples

### Content-Based Recommendations

```json
POST /api/v1/recommendations/content-based
{
  "recipe_id": "recipe_123",
  "num_recommendations": 10,
  "filters": {
    "max_calories": 500,
    "cuisine": "Mediterranean"
  }
}
```

### Collaborative Filtering

```json
POST /api/v1/recommendations/collaborative
{
  "user_id": "user_456",
  "num_recommendations": 10,
  "filters": {
    "min_protein": 20
  }
}
```

### Hybrid Recommendations

```json
POST /api/v1/recommendations/hybrid
{
  "user_id": "user_456",
  "num_recommendations": 10,
  "content_weight": 0.6,
  "collaborative_weight": 0.4
}
```

### Record User Interaction

```json
POST /api/v1/recommendations/interactions
{
  "user_id": "user_456",
  "recipe_id": "recipe_123",
  "interaction_type": "like",
  "rating": 5,
  "metadata": {
    "session_id": "session_789"
  }
}
```

## Configuration

Key environment variables:

```bash
# Database
MONGODB_URL=your_mongodb_connection_string
DATABASE_NAME=health_lover

# API
PROJECT_NAME=Health Lover Recommendation API
API_V1_STR=/api/v1

# CORS (for Next.js frontend)
BACKEND_CORS_ORIGINS=your_frontend_url

# ML Models
ML_MODELS_PATH=./app/ml_models

# Logging
LOG_LEVEL=INFO
```

## Integration with Next.js Frontend

The API is designed to be called from your Next.js application:

```typescript
// Example API call from Next.js
const getRecommendations = async (userId: string) => {
  const response = await fetch('http://localhost:8000/api/v1/recommendations/hybrid', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      num_recommendations: 10,
      content_weight: 0.6,
      collaborative_weight: 0.4
    })
  });
  
  return response.json();
};
```

## Model Training

> Notebook will be available soon.

The API includes sample models for testing. For production:

1. **Prepare your data**: Export recipe and user interaction data from your Next.js app
2. **Train models**: Use the ML model code in `app/ml_models/recommendation_engine.py`
3. **Deploy models**: Save trained models to the `ML_MODELS_PATH` directory
4. **Retrain periodically**: Use the `/retrain` endpoint to update models with new data

## Deployment

For production deployment:

1. **Use a production ASGI server**: Gunicorn with Uvicorn workers
2. **Set up proper environment**: Production database, logging, etc.
3. **Configure CORS**: Add your production frontend URL
4. **Monitor performance**: Set up logging and monitoring
5. **Scale horizontally**: Use load balancers for high traffic

## Support

This API is designed to work with the Health Lover Next.js application. For issues or questions:

1. Check the API documentation at `/docs`
2. Monitor logs for error details
3. Use the `/health` endpoint for system status
