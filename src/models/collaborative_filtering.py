import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
import logging
# from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
# from surprise.model_selection import train_test_split
# Note: Surprise library commented out due to Python 3.13 compatibility issues
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender:
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        """
        Initialize collaborative filtering recommender.
        
        Args:
            n_factors: Number of latent factors for matrix factorization
            random_state: Random state for reproducibility
        """
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd_model = None
        self.surprise_model = None
        self.user_item_matrix = None
        self.track_to_idx = {}
        self.idx_to_track = {}
        self.user_to_idx = {}
        self.idx_to_user = {}
        
    def prepare_data(self, interaction_df: pd.DataFrame, user_id: str, n_recommendations: int = 20, exclude_known: bool = True) -> List[Tuple[str, float]]:
        """
        Prepare interaction data for collaborative filtering and get recommendations for a user.

        Args:
            interaction_df: DataFrame with columns ['user_id', 'track_id', 'rating']
            user_id: User ID to get recommendations for
            n_recommendations: Number of recommendations to return
            exclude_known: Whether to exclude tracks the user has already interacted with

        Returns:
            List of (track_id, predicted_rating) tuples
        """
        logger.info("Preparing data for collaborative filtering...")

        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return []

        if self.surprise_model is None:
            logger.error("No model trained. Call train_surprise_model() first.")
            return []

        logger.info(f"Getting recommendations for user {user_id}...")

        user_idx = self.user_to_idx[user_id]
        recommendations = []

        # Get tracks user has already interacted with
        known_tracks = set()
        if exclude_known and self.user_item_matrix is not None:
            known_tracks = set(np.where(self.user_item_matrix[user_idx] > 0)[0])

        # Generate predictions for all tracks using sklearn model
        try:
            # Encode user_id
            user_encoded = self.user_encoder.transform([user_id])[0]

            for track_idx, track_id in self.idx_to_track.items():
                if exclude_known and track_idx in known_tracks:
                    continue

                try:
                    # Encode track_id
                    track_encoded = self.track_encoder.transform([track_id])[0]

                    # Predict rating
                    prediction = self.surprise_model.predict([[user_encoded, track_encoded]])[0]
                    recommendations.append((track_id, float(prediction)))
                except:
                    # Skip tracks not seen during training
                    continue

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []

        # Sort by predicted rating and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations[:n_recommendations]
    
    def get_similar_tracks(self, track_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Get tracks similar to a given track using item-based collaborative filtering.
        
        Args:
            track_id: Track ID to find similar tracks for
            n_similar: Number of similar tracks to return
            
        Returns:
            List of (similar_track_id, similarity_score) tuples
        """
        if track_id not in self.track_to_idx:
            logger.warning(f"Track {track_id} not found in training data")
            return []
        
        if self.user_item_matrix is None:
            logger.error("No data prepared. Call prepare_data() first.")
            return []
        
        logger.info(f"Finding tracks similar to {track_id}...")
        
        track_idx = self.track_to_idx[track_id]
        
        # Get the track's rating vector
        track_vector = self.user_item_matrix[:, track_idx].reshape(1, -1)
        
        # Calculate similarity with all other tracks
        similarities = cosine_similarity(track_vector, self.user_item_matrix.T)[0]
        
        # Get similar tracks (excluding the track itself)
        similar_tracks = []
        for idx, similarity in enumerate(similarities):
            if idx != track_idx and similarity > 0:
                similar_track_id = self.idx_to_track[idx]
                similar_tracks.append((similar_track_id, similarity))
        
        # Sort by similarity and return top N
        similar_tracks.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(similar_tracks)} similar tracks")
        return similar_tracks[:n_similar]
    
    def get_user_embeddings(self) -> np.ndarray:
        """Get user embeddings from trained SVD model."""
        if self.svd_model is None:
            logger.error("SVD model not trained")
            return np.array([])
        
        return self.svd_model.transform(self.user_item_matrix)
    
    def get_item_embeddings(self) -> np.ndarray:
        """Get item (track) embeddings from trained SVD model."""
        if self.svd_model is None:
            logger.error("SVD model not trained")
            return np.array([])
        
        return self.svd_model.components_.T
    
    def explain_recommendation(self, user_id: str, track_id: str, 
                             top_k_factors: int = 3) -> Dict[str, any]:
        """
        Explain why a track was recommended to a user.
        
        Args:
            user_id: User ID
            track_id: Track ID
            top_k_factors: Number of top factors to include in explanation
            
        Returns:
            Dictionary with explanation details
        """
        if user_id not in self.user_to_idx or track_id not in self.track_to_idx:
            return {"error": "User or track not found"}
        
        if self.surprise_model is None:
            return {"error": "No model trained"}
        
        # Get prediction
        prediction = self.surprise_model.predict(user_id, track_id)
        
        explanation = {
            "predicted_rating": prediction.est,
            "user_id": user_id,
            "track_id": track_id,
            "was_impossible": prediction.details["was_impossible"]
        }
        
        # Get user and item factors if available
        if hasattr(self.surprise_model, 'pu') and hasattr(self.surprise_model, 'qi'):
            user_factors = self.surprise_model.pu[self.surprise_model.trainset.to_inner_uid(user_id)]
            item_factors = self.surprise_model.qi[self.surprise_model.trainset.to_inner_iid(track_id)]
            
            # Calculate factor contributions
            factor_contributions = user_factors * item_factors
            top_factor_indices = np.argsort(np.abs(factor_contributions))[-top_k_factors:]
            
            explanation["top_factors"] = {
                f"factor_{idx}": float(factor_contributions[idx]) 
                for idx in reversed(top_factor_indices)
            }
        
        return explanation


class ImplicitCollaborativeFiltering:
    """
    Collaborative filtering for implicit feedback using the implicit library.
    Useful when we only have play counts, not explicit ratings.
    """
    
    def __init__(self, factors: int = 50, regularization: float = 0.01, 
                 iterations: int = 20, alpha: float = 40.0):
        """
        Initialize implicit collaborative filtering model.
        
        Args:
            factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of training iterations
            alpha: Confidence scaling parameter
        """
        try:
            from implicit.als import AlternatingLeastSquares
            self.model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                alpha=alpha,
                random_state=42
            )
            self.available = True
        except ImportError:
            logger.warning("implicit library not available. Install with: pip install implicit")
            self.available = False
        
        self.user_tracks = None
        self.track_users = None
        self.user_to_idx = {}
        self.track_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_track = {}
    
    def prepare_implicit_data(self, interaction_df: pd.DataFrame) -> None:
        """
        Prepare data for implicit feedback model.
        
        Args:
            interaction_df: DataFrame with user-track interactions
        """
        if not self.available:
            logger.error("Implicit library not available")
            return
        
        logger.info("Preparing data for implicit collaborative filtering...")
        
        # Create mappings
        unique_users = interaction_df['user_id'].unique()
        unique_tracks = interaction_df['track_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.track_to_idx = {track: idx for idx, track in enumerate(unique_tracks)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_track = {idx: track for track, idx in self.track_to_idx.items()}
        
        # Create sparse matrix
        rows = [self.user_to_idx[user] for user in interaction_df['user_id']]
        cols = [self.track_to_idx[track] for track in interaction_df['track_id']]
        data = interaction_df['rating'].values
        
        self.user_tracks = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(unique_users), len(unique_tracks))
        )
        self.track_users = self.user_tracks.T.tocsr()
        
        logger.info(f"Created sparse matrix: {self.user_tracks.shape}")
    
    def train(self) -> None:
        """Train the implicit ALS model."""
        if not self.available or self.user_tracks is None:
            logger.error("Cannot train: data not prepared or library not available")
            return
        
        logger.info("Training implicit ALS model...")
        self.model.fit(self.user_tracks)
        logger.info("Implicit ALS model training completed")
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 20) -> List[Tuple[str, float]]:
        """Get recommendations for a user using implicit model."""
        if not self.available or user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get recommendations
        track_indices, scores = self.model.recommend(
            user_idx, 
            self.user_tracks[user_idx], 
            N=n_recommendations
        )
        
        recommendations = [
            (self.idx_to_track[track_idx], float(score))
            for track_idx, score in zip(track_indices, scores)
        ]
        
        return recommendations
        # Create mappings
        unique_tracks = interaction_df['track_id'].unique()
        unique_users = interaction_df['user_id'].unique()
        
        self.track_to_idx = {track: idx for idx, track in enumerate(unique_tracks)}
        self.idx_to_track = {idx: track for track, idx in self.track_to_idx.items()}
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        # Create user-item matrix
        n_users = len(unique_users)
        n_tracks = len(unique_tracks)
        
        self.user_item_matrix = np.zeros((n_users, n_tracks))
        
        for _, row in interaction_df.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            track_idx = self.track_to_idx[row['track_id']]
            self.user_item_matrix[user_idx, track_idx] = row['rating']
        
        logger.info(f"Created user-item matrix: {n_users} users x {n_tracks} tracks")
    
    def train_svd_model(self) -> None:
        """Train SVD matrix factorization model."""
        if self.user_item_matrix is None:
            logger.error("No data prepared. Call prepare_data() first.")
            return
        
        logger.info("Training SVD model...")
        
        # Use TruncatedSVD for sparse matrices
        self.svd_model = TruncatedSVD(
            n_components=min(self.n_factors, min(self.user_item_matrix.shape) - 1),
            random_state=self.random_state
        )
        
        # Fit the model
        self.svd_model.fit(self.user_item_matrix)
        
        logger.info("SVD model training completed")
    
    def train_surprise_model(self, interaction_df: pd.DataFrame) -> float:
        """
        Train scikit-learn based SVD model (alternative to Surprise).
        
        Args:
            interaction_df: Interaction data
            
        Returns:
            RMSE score on test set
        """
        if interaction_df.empty:
            logger.error("Empty interaction data provided")
            return float('inf')
        
        logger.info("Training scikit-learn SVD model...")
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            
            # Prepare data for sklearn
            X = interaction_df[['user_id', 'track_id']].copy()
            y = interaction_df['rating'].values
            
            # Convert categorical to numeric
            from sklearn.preprocessing import LabelEncoder
            user_encoder = LabelEncoder()
            track_encoder = LabelEncoder()
            
            X['user_encoded'] = user_encoder.fit_transform(X['user_id'])
            X['track_encoded'] = track_encoder.fit_transform(X['track_id'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X[['user_encoded', 'track_encoded']], y, 
                test_size=0.2, random_state=self.random_state
            )
            
            # Train simple linear model as baseline
            from sklearn.linear_model import LinearRegression
            self.surprise_model = LinearRegression()
            self.surprise_model.fit(X_train, y_train)
            
            # Store encoders for prediction
            self.user_encoder = user_encoder
            self.track_encoder = track_encoder
            
            # Evaluate
            y_pred = self.surprise_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            logger.info(f"Scikit-learn model trained with RMSE: {rmse:.4f}")
            return rmse
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return float('inf')
    
    def get_user_recommendations(self, user_id: str, n_recommendations: int = 20, 
                               exclude_known: bool = True) -> List[Tuple[str, float]]:
        """
        Get recommendations for a user using collaborative filtering.
        
        Args:
            user_id: User ID to get recommendations for
            n_recommendations: Number of recommendations to return
            exclude_known: Whether to exclude tracks the user has already interacted with
            
        Returns:
            List of (track_id, predicted_rating) tuples
        """
        if self.svd_model is None:
            logger.error("SVD model is not trained")
            return []

        user_idx = self.user_encoder.transform([user_id])[0]
        user_ratings = self.svd_model.transform(self.user_item_matrix[user_idx:user_idx+1])
        track_ids = self.track_encoder.inverse_transform(np.arange(self.n_tracks))

        # Create a list of (track_id, predicted_rating) tuples
        recommendations = list(zip(track_ids, user_ratings.flatten()))

        # Exclude known tracks
        if exclude_known:
            known_tracks = set(self.user_item_matrix[user_idx].nonzero()[1])
            recommendations = [rec for rec in recommendations if rec[0] not in known_tracks]

        # Sort by predicted rating
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:n_recommendations]