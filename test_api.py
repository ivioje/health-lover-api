"""
Quick test script to verify the FastAPI recommendation service
"""

import requests
import json
import sys

def test_api():
    BASE_URL = "http://localhost:8000"
    
    print("🧪 Testing Health Lover Recommendation API...")
    print(f"Base URL: {BASE_URL}")
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/recommendations/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Content-based recommendations
    print("\n2. Testing content-based recommendations...")
    try:
        payload = {
            "user_id": "test_user_123",
            "preferences": {
                "dietary_restrictions": ["gluten-free"],
                "health_goals": ["weight-loss"],
                "activity_level": "moderate",
                "preferred_cuisines": ["mediterranean"],
                "disliked_ingredients": ["mushrooms"]
            },
            "num_recommendations": 5
        }
        
        response = requests.post(f"{BASE_URL}/recommendations/content-based", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Content-based recommendations: {len(data['recommendations'])} recipes")
            if data['recommendations']:
                print(f"   Sample: {data['recommendations'][0]['recipe']}")
        else:
            print(f"❌ Content-based recommendations failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Content-based recommendations error: {e}")
    
    # Test 3: Trending recipes
    print("\n3. Testing trending recipes...")
    try:
        response = requests.get(f"{BASE_URL}/recommendations/trending?num_recommendations=3")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Trending recipes: {len(data['trending_recipes'])} recipes")
            if data['trending_recipes']:
                print(f"   Sample: {data['trending_recipes'][0]['recipe']}")
        else:
            print(f"❌ Trending recipes failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Trending recipes error: {e}")
    
    # Test 4: System stats
    print("\n4. Testing system stats...")
    try:
        response = requests.get(f"{BASE_URL}/recommendations/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System stats: {data.get('total_recipes', 0)} total recipes")
        else:
            print(f"❌ System stats failed: {response.status_code}")
    except Exception as e:
        print(f"❌ System stats error: {e}")
    
    print("\n🎉 API testing completed!")
    return True

if __name__ == "__main__":
    test_api()
