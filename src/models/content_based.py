import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self, similarity_metric: str = 'cosine'):
        """
        Initialize content-based recommender.
        
        Args:
            similarity_metric: Similarity metric to use ('cosine', 'euclidean')
        """
        self.similarity_metric = similarity_metric
        self.scaler = StandardScaler()
        self.audio_features = None
        self.similarity_matrix = None
        self.track_to_idx = {}
        self.idx_to_track = {}
        self.feature_columns = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
        ]
        self.user_profile = None
        self.track_clusters = None
        
    def prepare_features(self, audio_features_df: pd.DataFrame) -> None:
        """
        Prepare and normalize audio features.
        
        Args:
            audio_features_df: DataFrame with audio features
        """
        logger.info("Preparing audio features for content-based filtering...")
        
        if audio_features_df.empty:
            logger.warning("Empty audio features provided")
            return
        
        # Store the features
        self.audio_features = audio_features_df.copy()
        
        # Create track mappings
        self.track_to_idx = {
            track: idx for idx, track in enumerate(self.audio_features['track_id'])
        }
        self.idx_to_track = {
            idx: track for track, idx in self.track_to_idx.items()
        }
        
        # Normalize features
        available_features = [col for col in self.feature_columns if col in self.audio_features.columns]
        
        if available_features:
            feature_matrix = self.audio_features[available_features].values
            normalized_features = self.scaler.fit_transform(feature_matrix)
            
            # Update the dataframe with normalized features
            for i, col in enumerate(available_features):
                self.audio_features[f'{col}_norm'] = normalized_features[:, i]
        
        logger.info(f"Prepared features for {len(self.audio_features)} tracks")
    
    def compute_similarity_matrix(self) -> None:
        """Compute track similarity matrix based on audio features."""
        if self.audio_features is None:
            logger.error("Audio features not prepared. Call prepare_features() first.")
            return
        
        logger.info("Computing similarity matrix...")
        
        # Get normalized feature columns
        feature_cols = [col for col in self.audio_features.columns if col.endswith('_norm')]
        
        if not feature_cols:
            logger.error("No normalized features found")
            return
        
        feature_matrix = self.audio_features[feature_cols].values
        
        # Compute similarity matrix
        if self.similarity_metric == 'cosine':
            self.similarity_matrix = cosine_similarity(feature_matrix)
        elif self.similarity_metric == 'euclidean':
            # Convert distance to similarity (1 / (1 + distance))
            distances = euclidean_distances(feature_matrix)
            self.similarity_matrix = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        logger.info(f"Computed {self.similarity_matrix.shape} similarity matrix")
    
    def build_user_profile(self, user_tracks: List[str], user_ratings: Optional[List[float]] = None) -> None:
        """
        Build user profile based on their listening history.
        
        Args:
            user_tracks: List of track IDs the user has listened to
            user_ratings: Optional ratings for each track (if None, equal weights used)
        """
        logger.info("Building user profile...")
        
        if self.audio_features is None:
            logger.error("Audio features not prepared")
            return
        
        # Filter tracks that exist in our feature data
        valid_tracks = [track for track in user_tracks if track in self.track_to_idx]
        
        if not valid_tracks:
            logger.warning("No valid tracks found in user history")
            return
        
        # Get feature matrix for user tracks
        track_indices = [self.track_to_idx[track] for track in valid_tracks]
        feature_cols = [col for col in self.audio_features.columns if col.endswith('_norm')]
        
        if not feature_cols:
            logger.error("No normalized features available")
            return
        
        user_track_features = self.audio_features.iloc[track_indices][feature_cols].values
        
        # Create weighted profile
        if user_ratings is None:
            user_ratings = np.ones(len(valid_tracks))
        else:
            user_ratings = np.array(user_ratings[:len(valid_tracks)])
        
        # Normalize ratings
        if len(user_ratings) > 1:
            user_ratings = (user_ratings - user_ratings.min()) / (user_ratings.max() - user_ratings.min() + 1e-8)
        
        # Compute weighted average profile
        self.user_profile = np.average(user_track_features, axis=0, weights=user_ratings)
        
        logger.info(f"Built user profile from {len(valid_tracks)} tracks")
    
    def get_track_recommendations(self, user_tracks: List[str], 
                                user_ratings: Optional[List[float]] = None,
                                n_recommendations: int = 20,
                                exclude_known: bool = True) -> List[Tuple[str, float]]:
        """
        Get content-based recommendations for a user.
        
        Args:
            user_tracks: List of tracks the user has listened to
            user_ratings: Optional ratings for user tracks
            n_recommendations: Number of recommendations to return
            exclude_known: Whether to exclude tracks the user already knows
            
        Returns:
            List of (track_id, similarity_score) tuples
        """
        logger.info("Generating content-based recommendations...")
        
        # Build user profile
        self.build_user_profile(user_tracks, user_ratings)
        
        if self.user_profile is None:
            logger.error("Failed to build user profile")
            return []
        
        # Get feature matrix
        feature_cols = [col for col in self.audio_features.columns if col.endswith('_norm')]
        all_track_features = self.audio_features[feature_cols].values
        
        # Compute similarity between user profile and all tracks
        similarities = cosine_similarity([self.user_profile], all_track_features)[0]
        
        # Create recommendations
        recommendations = []
        known_tracks = set(user_tracks) if exclude_known else set()
        
        for idx, similarity in enumerate(similarities):
            track_id = self.idx_to_track[idx]
            if track_id not in known_tracks:
                recommendations.append((track_id, float(similarity)))
        
        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} content-based recommendations")
        return recommendations[:n_recommendations]
    
    def get_similar_tracks(self, track_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Get tracks similar to a given track based on audio features.
        
        Args:
            track_id: Track ID to find similar tracks for
            n_similar: Number of similar tracks to return
            
        Returns:
            List of (similar_track_id, similarity_score) tuples
        """
        if track_id not in self.track_to_idx:
            logger.warning(f"Track {track_id} not found")
            return []
        
        if self.similarity_matrix is None:
            logger.error("Similarity matrix not computed. Call compute_similarity_matrix() first.")
            return []
        
        track_idx = self.track_to_idx[track_id]
        similarities = self.similarity_matrix[track_idx]
        
        # Get similar tracks (excluding the track itself)
        similar_tracks = []
        for idx, similarity in enumerate(similarities):
            if idx != track_idx:
                similar_track_id = self.idx_to_track[idx]
                similar_tracks.append((similar_track_id, float(similarity)))
        
        # Sort by similarity and return top N
        similar_tracks.sort(key=lambda x: x[1], reverse=True)
        
        return similar_tracks[:n_similar]
    
    def cluster_tracks(self, n_clusters: int = 10) -> Dict[str, int]:
        """
        Cluster tracks based on audio features.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping track_id to cluster_id
        """
        logger.info(f"Clustering tracks into {n_clusters} clusters...")
        
        if self.audio_features is None:
            logger.error("Audio features not prepared")
            return {}
        
        # Get normalized features
        feature_cols = [col for col in self.audio_features.columns if col.endswith('_norm')]
        feature_matrix = self.audio_features[feature_cols].values
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # Create track to cluster mapping
        self.track_clusters = {}
        for idx, cluster_id in enumerate(cluster_labels):
            track_id = self.idx_to_track[idx]
            self.track_clusters[track_id] = int(cluster_id)
        
        logger.info("Track clustering completed")
        return self.track_clusters
    
    def get_cluster_recommendations(self, user_tracks: List[str], 
                                  n_recommendations: int = 20) -> List[Tuple[str, float]]:
        """
        Get recommendations based on track clusters.
        
        Args:
            user_tracks: User's track history
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended tracks with cluster-based scores
        """
        if self.track_clusters is None:
            self.cluster_tracks()
        
        # Find user's preferred clusters
        user_clusters = []
        for track in user_tracks:
            if track in self.track_clusters:
                user_clusters.append(self.track_clusters[track])
        
        if not user_clusters:
            return []
        
        # Count cluster preferences
        from collections import Counter
        cluster_counts = Counter(user_clusters)
        total_tracks = len(user_clusters)
        
        # Get recommendations from preferred clusters
        recommendations = []
        known_tracks = set(user_tracks)
        
        for track_id, cluster_id in self.track_clusters.items():
            if track_id not in known_tracks and cluster_id in cluster_counts:
                # Score based on cluster preference
                cluster_score = cluster_counts[cluster_id] / total_tracks
                recommendations.append((track_id, cluster_score))
        
        # Sort and return
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def analyze_user_preferences(self, user_tracks: List[str]) -> Dict[str, float]:
        """
        Analyze user's musical preferences based on their track history.
        
        Args:
            user_tracks: List of user's tracks
            
        Returns:
            Dictionary with preference analysis
        """
        if self.audio_features is None:
            return {}
        
        # Get valid tracks
        valid_tracks = [track for track in user_tracks if track in self.track_to_idx]
        
        if not valid_tracks:
            return {}
        
        # Get audio features for user tracks
        track_indices = [self.track_to_idx[track] for track in valid_tracks]
        user_features = self.audio_features.iloc[track_indices]
        
        # Calculate preference scores
        preferences = {}
        
        for feature in self.feature_columns:
            if feature in user_features.columns:
                mean_val = user_features[feature].mean()
                std_val = user_features[feature].std()
                preferences[f'avg_{feature}'] = float(mean_val)
                preferences[f'std_{feature}'] = float(std_val)
        
        # Add derived preferences
        if 'energy' in user_features.columns and 'valence' in user_features.columns:
            preferences['mood_score'] = float(
                (user_features['energy'] * user_features['valence']).mean()
            )
        
        if 'danceability' in user_features.columns and 'tempo' in user_features.columns:
            preferences['dance_score'] = float(
                (user_features['danceability'] * user_features['tempo']).mean()
            )
        
        return preferences
    
    def get_feature_importance(self, user_tracks: List[str], 
                             reference_tracks: List[str]) -> Dict[str, float]:
        """
        Calculate which audio features are most important for user preferences.
        
        Args:
            user_tracks: User's liked tracks
            reference_tracks: Reference tracks for comparison
            
        Returns:
            Dictionary with feature importance scores
        """
        if self.audio_features is None:
            return {}
        
        # Get features for both sets
        user_indices = [self.track_to_idx[t] for t in user_tracks if t in self.track_to_idx]
        ref_indices = [self.track_to_idx[t] for t in reference_tracks if t in self.track_to_idx]
        
        if not user_indices or not ref_indices:
            return {}
        
        user_features = self.audio_features.iloc[user_indices]
        ref_features = self.audio_features.iloc[ref_indices]
        
        # Calculate feature importance as difference in means
        importance = {}
        for feature in self.feature_columns:
            if feature in user_features.columns:
                user_mean = user_features[feature].mean()
                ref_mean = ref_features[feature].mean()
                importance[feature] = abs(user_mean - ref_mean)
        
        # Normalize importance scores
        if importance:
            max_importance = max(importance.values())
            importance = {k: v/max_importance for k, v in importance.items()}
        
        return importance