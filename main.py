import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# --- GLOBAL VARIABLES & DATA LOADING ---
df = None
user_item_matrix = None
model_knn = None
item_ids = []

# --- UTILITY FUNCTION TO LOAD DATA ---
def load_model_and_data():
    global df, user_item_matrix, model_knn, item_ids
    print("Loading and preparing data...")
    try:
        # Load the clothing ratings dataset from the provided CSV file.
        df = pd.read_csv("clothing_dataset.csv")

        # Create the user-item matrix for collaborative filtering
        user_item_matrix = df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)

        # Store the list of all available item IDs
        item_ids = user_item_matrix.columns.tolist()

        # Initialize and fit the NearestNeighbors model
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(user_item_matrix.T)

        print("Data and model loaded successfully!")

    except FileNotFoundError:
        print("Error: The file 'clothing_dataset.csv' was not found.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit(1)


# --- APPLICATION LIFESPAN CONTEXT MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_data()
    yield


# --- REQUEST BODY MODEL ---
class RecommendationRequest(BaseModel):
    item_id: str
    num_recommendations: int = 5


# --- RECOMMENDATION LOGIC ---
def recommend_items(item_name: str, num_recommendations: int) -> List[str]:
    if item_name not in user_item_matrix.columns:
        return []
    item_index = user_item_matrix.columns.get_loc(item_name)
    item_vector = user_item_matrix.iloc[:, item_index].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(item_vector, n_neighbors=num_recommendations + 1)
    recommended_item_indices = indices.flatten()[1:]
    recommended_item_names = [user_item_matrix.columns[i] for i in recommended_item_indices]
    return recommended_item_names


# --- FastAPI app instance ---
app = FastAPI(
    title="Clothing Recommendation API",
    description="An API to get clothing recommendations based on collaborative filtering.",
    lifespan=lifespan
)


# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    return {
        "message": "Clothing Recommendation API is running 🚀",
        "docs_url": "/docs",
        "recommend_endpoint": "/recommend",
        "health_check": "/health"
    }


# --- HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# --- RECOMMENDATION ENDPOINT ---
@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    global user_item_matrix

    if user_item_matrix is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded. Please try again later.")

    if request.item_id not in user_item_matrix.columns:
        raise HTTPException(status_code=404, detail=f"Item ID '{request.item_id}' not found.")

    recommendations = recommend_items(
        item_name=request.item_id,
        num_recommendations=request.num_recommendations
    )

    return {"item_id": request.item_id, "recommendations": recommendations}


