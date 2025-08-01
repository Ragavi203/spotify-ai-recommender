import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyDataProcessor:
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = StandardScaler()
        self.feature_columns = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
        ]
        
    def create_user_item_matrix(self, user_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create user-item interaction matrix from user data.
        
        Args:
            user_data: Dictionary containing user's Spotify data
            
        Returns:
            User-item matrix with implicit ratings
        """
        logger.info("Creating user-item interaction matrix...")
        
        # Combine all track interactions
        interactions = []
        
        # Process top tracks (higher weight for more recent)
        for time_range, weight in [('short_term', 3), ('medium_term', 2), ('long_term', 1)]:
            key = f'top_tracks_{time_range}'
            if key in user_data and not user_data[key].empty:
                df = user_data[key].copy()
                df['rating'] = weight * (51 - df.index)  # Higher rating for higher ranked tracks
                df['user_id'] = 'current_user'
                interactions.append(df[['user_id', 'track_id', 'rating']])
        
        # Process recently played (frequency-based rating)
        if 'recently_played' in user_data and not user_data['recently_played'].empty:
            recent_df = user_data['recently_played'].copy()
            # Count frequency of plays
            play_counts = recent_df['track_id'].value_counts()
            
            for track_id, count in play_counts.items():
                interactions.append(pd.DataFrame({
                    'user_id': ['current_user'],
                    'track_id': [track_id],
                    'rating': [count * 0.5]  # Weight recent plays
                }))
        
        # Combine all interactions
        if interactions:
            interaction_matrix = pd.concat(interactions, ignore_index=True)
            
            # Aggregate ratings for same track
            interaction_matrix = interaction_matrix.groupby(['user_id', 'track_id'])['rating'].sum().reset_index()
            
            # Normalize ratings to 1-5 scale
            max_rating = interaction_matrix['rating'].max()
            interaction_matrix['rating'] = (interaction_matrix['rating'] / max_rating) * 4 + 1
            
            logger.info(f"Created interaction matrix with {len(interaction_matrix)} interactions")
            return interaction_matrix
        
        return pd.DataFrame()
    
    def process_audio_features(self, audio_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and normalize audio features.
        
        Args:
            audio_features_df: Raw audio features from Spotify API
            
        Returns:
            Processed and normalized audio features
        """
        if audio_features_df.empty:
            return pd.DataFrame()
        
        logger.info("Processing audio features...")
        
        # Create a copy to avoid modifying original
        features_df = audio_features_df.copy()
        
        # Handle missing values
        features_df = features_df.dropna(subset=self.feature_columns)
        
        # Create categorical features
        features_df['key_mode'] = features_df['key'].astype(str) + '_' + features_df['mode'].astype(str)
        features_df['time_signature_cat'] = features_df['time_signature'].astype(str)
        
        # Normalize continuous features
        continuous_features = [col for col in self.feature_columns if col in features_df.columns]
        
        if continuous_features:
            features_df[continuous_features] = self.scaler.fit_transform(features_df[continuous_features])
        
        # Create derived features
        features_df['energy_valence'] = features_df['energy'] * features_df['valence']
        features_df['danceability_tempo'] = features_df['danceability'] * features_df['tempo']
        features_df['acoustic_energy'] = features_df['acousticness'] * (1 - features_df['energy'])
        
        logger.info(f"Processed audio features for {len(features_df)} tracks")
        return features_df
    
    def create_content_similarity_matrix(self, audio_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create content-based similarity matrix using audio features.
        
        Args:
            audio_features_df: Processed audio features
            
        Returns:
            Track similarity matrix
        """
        if audio_features_df.empty:
            return pd.DataFrame()
        
        logger.info("Creating content similarity matrix...")
        
        # Select features for similarity calculation
        feature_cols = [col for col in self.feature_columns if col in audio_features_df.columns]
        feature_cols.extend(['energy_valence', 'danceability_tempo', 'acoustic_energy'])
        
        # Get feature matrix
        feature_matrix = audio_features_df[feature_cols].values
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(feature_matrix)
        
        # Create DataFrame with track IDs as index and columns
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=audio_features_df['track_id'],
            columns=audio_features_df['track_id']
        )
        
        logger.info(f"Created similarity matrix for {len(similarity_df)} tracks")
        return similarity_df
    
    def extract_user_preferences(self, user_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract user preferences from listening history.
        
        Args:
            user_data: User's Spotify data
            
        Returns:
            Dictionary of user preferences
        """
        logger.info("Extracting user preferences...")
        
        preferences = {}
        
        # Combine all user tracks with weights
        all_tracks = []
        weights = []
        
        # Weight recent tracks more heavily
        for time_range, weight in [('short_term', 3), ('medium_term', 2), ('long_term', 1)]:
            key = f'top_tracks_{time_range}'
            if key in user_data and not user_data[key].empty:
                tracks = user_data[key]
                all_tracks.append(tracks)
                weights.extend([weight] * len(tracks))
        
        if not all_tracks:
            return preferences
        
        combined_tracks = pd.concat(all_tracks, ignore_index=True)
        
        # Get audio features for preference calculation
        if 'audio_features' in user_data and not user_data['audio_features'].empty:
            audio_features = user_data['audio_features']
            
            # Merge tracks with audio features
            track_features = combined_tracks.merge(
                audio_features, on='track_id', how='inner'
            )
            
            if not track_features.empty:
                # Calculate weighted averages for each audio feature
                weights_array = np.array(weights[:len(track_features)])
                
                for feature in self.feature_columns:
                    if feature in track_features.columns:
                        weighted_avg = np.average(
                            track_features[feature], 
                            weights=weights_array
                        )
                        preferences[f'avg_{feature}'] = weighted_avg
                
                # Calculate preference distributions
                preferences['high_energy_preference'] = (track_features['energy'] > 0.7).mean()
                preferences['high_valence_preference'] = (track_features['valence'] > 0.7).mean()
                preferences['acoustic_preference'] = (track_features['acousticness'] > 0.5).mean()
                preferences['dance_preference'] = (track_features['danceability'] > 0.7).mean()
        
        # Extract genre and artist preferences (if available)
        artist_counts = combined_tracks['artist_name'].value_counts()
        preferences['top_artists'] = artist_counts.head(10).to_dict()
        
        # Extract popularity preferences
        if 'popularity' in combined_tracks.columns:
            preferences['avg_popularity'] = np.average(
                combined_tracks['popularity'], 
                weights=weights[:len(combined_tracks)]
            )
        
        logger.info("User preferences extracted successfully")
        return preferences
    
    def prepare_recommendation_data(self, user_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Prepare all data needed for recommendation algorithms.
        
        Args:
            user_data: Raw user data from Spotify
            
        Returns:
            Dictionary containing processed data for recommendations
        """
        logger.info("Preparing recommendation data...")
        
        processed_data = {}
        
        # Create interaction matrix
        processed_data['interaction_matrix'] = self.create_user_item_matrix(user_data)
        
        # Process audio features
        if 'audio_features' in user_data:
            processed_data['audio_features'] = self.process_audio_features(user_data['audio_features'])
            processed_data['similarity_matrix'] = self.create_content_similarity_matrix(
                processed_data['audio_features']
            )
        
        # Extract user preferences
        processed_data['user_preferences'] = self.extract_user_preferences(user_data)
        
        # Store original data for reference
        processed_data['raw_data'] = user_data
        
        logger.info("Recommendation data preparation completed")
        return processed_data
    
    def get_track_features_vector(self, track_id: str, audio_features_df: pd.DataFrame) -> np.ndarray:
        """
        Get feature vector for a specific track.
        
        Args:
            track_id: Spotify track ID
            audio_features_df: Processed audio features DataFrame
            
        Returns:
            Feature vector for the track
        """
        if track_id not in audio_features_df['track_id'].values:
            return np.array([])
        
        track_row = audio_features_df[audio_features_df['track_id'] == track_id]
        feature_cols = [col for col in self.feature_columns if col in audio_features_df.columns]
        
        return track_row[feature_cols].values.flatten()
    
    def calculate_track_diversity(self, track_ids: List[str], audio_features_df: pd.DataFrame) -> float:
        """
        Calculate diversity score for a list of tracks.
        
        Args:
            track_ids: List of track IDs
            audio_features_df: Audio features DataFrame
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(track_ids) < 2:
            return 0.0
        
        # Get feature vectors for all tracks
        vectors = []
        for track_id in track_ids:
            vector = self.get_track_features_vector(track_id, audio_features_df)
            if len(vector) > 0:
                vectors.append(vector)
        
        if len(vectors) < 2:
            return 0.0
        
        # Calculate pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(vectors)
        
        # Return average pairwise distance (excluding diagonal)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        return distances[mask].mean()