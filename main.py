# main.py
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# --- GLOBAL VARIABLES & DATA LOADING ---
# These variables will be loaded once when the application starts.
df = None
user_item_matrix = None
model_knn = None
item_ids = []

# --- UTILITY FUNCTION TO LOAD DATA ---
def load_model_and_data():
    """
    Loads the dataset and fits the NearestNeighbors model.
    This function is called once on application startup.
    """
    global df, user_item_matrix, model_knn, item_ids
    print("Loading and preparing data...")
    try:
        # Load the clothing ratings dataset from the provided CSV file.
        # Note: You must ensure 'clothing_dataset.csv' is in the same directory.
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
        model_knn.fit(user_item_matrix.T) # Transpose to find similar items based on user ratings

        print("Data and model loaded successfully!")

    except FileNotFoundError:
        print("Error: The file 'clothing_dataset.csv' was not found.")
        print("Please make sure the file is in the same directory as the script.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit(1)


# --- APPLICATION LIFESPAN CONTEXT MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    The 'yield' statement separates startup logic from shutdown logic.
    """
    # Startup logic
    load_model_and_data()
    yield
    # Shutdown logic (none needed for this simple app)


# --- REQUEST BODY MODEL ---
class RecommendationRequest(BaseModel):
    item_id: str
    num_recommendations: int = 5


# --- RECOMMENDATION LOGIC ---
def recommend_items(item_name: str, num_recommendations: int) -> List[str]:
    """
    Recommends similar items based on the provided item_name using the KNN model.
    """
    # Check if the requested item exists in our dataset
    if item_name not in user_item_matrix.columns:
        return []

    # Get the item's vector
    item_index = user_item_matrix.columns.get_loc(item_name)
    item_vector = user_item_matrix.iloc[:, item_index].values.reshape(1, -1)

    # Find the nearest neighbors (similar items)
    # The first neighbor is always the item itself, so we request one more.
    distances, indices = model_knn.kneighbors(item_vector, n_neighbors=num_recommendations + 1)

    # Extract the indices of the recommended items, skipping the first one
    recommended_item_indices = indices.flatten()[1:]

    # Get the actual item IDs from the indices
    recommended_item_names = [user_item_matrix.columns[i] for i in recommended_item_indices]

    return recommended_item_names


# --- FastAPI app instance ---
# Pass the lifespan context manager to the FastAPI instance
app = FastAPI(
    title="Clothing Recommendation API",
    description="An API to get clothing recommendations based on collaborative filtering.",
    lifespan=lifespan
)


# --- API ENDPOINT ---
@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """
    Endpoint to get recommendations for a given item.

    - **item_id**: The ID of the item you want recommendations for (e.g., 'Blue Denim Jeans').
    - **num_recommendations**: The number of recommendations to return.
    """
    global user_item_matrix
    
    # Check if the model has been loaded
    if user_item_matrix is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded. Please try again in a moment.")

    # Validate that the requested item exists in our dataset
    if request.item_id not in user_item_matrix.columns:
        raise HTTPException(status_code=404, detail=f"Item ID '{request.item_id}' not found.")

    # Get the recommendations
    recommendations = recommend_items(
        item_name=request.item_id,
        num_recommendations=request.num_recommendations
    )


    return {"item_id": request.item_id, "recommendations": recommendations}
