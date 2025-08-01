import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyClient:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize Spotify client with OAuth authentication."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        
        # Set up authentication with required scopes
        scope = "user-read-recently-played user-top-read user-library-read playlist-read-private"
        
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope
        ))
        
        logger.info("Spotify client initialized successfully")
    
    def get_user_top_tracks(self, limit: int = 50, time_range: str = 'medium_term') -> pd.DataFrame:
        """
        Get user's top tracks.
        
        Args:
            limit: Number of tracks to retrieve (max 50)
            time_range: 'short_term', 'medium_term', or 'long_term'
        """
        try:
            results = self.sp.current_user_top_tracks(limit=limit, time_range=time_range)
            tracks_data = []
            
            for item in results['items']:
                track_info = {
                    'track_id': item['id'],
                    'track_name': item['name'],
                    'artist_name': item['artists'][0]['name'],
                    'album_name': item['album']['name'],
                    'popularity': item['popularity'],
                    'duration_ms': item['duration_ms'],
                    'explicit': item['explicit'],
                    'release_date': item['album']['release_date']
                }
                tracks_data.append(track_info)
            
            logger.info(f"Retrieved {len(tracks_data)} top tracks")
            return pd.DataFrame(tracks_data)
            
        except Exception as e:
            logger.error(f"Error fetching top tracks: {e}")
            return pd.DataFrame()
    
    def get_recently_played(self, limit: int = 50) -> pd.DataFrame:
        """Get user's recently played tracks."""
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            tracks_data = []
            
            for item in results['items']:
                track = item['track']
                track_info = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artist_name': track['artists'][0]['name'],
                    'album_name': track['album']['name'],
                    'played_at': item['played_at'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                }
                tracks_data.append(track_info)
            
            logger.info(f"Retrieved {len(tracks_data)} recently played tracks")
            return pd.DataFrame(tracks_data)
            
        except Exception as e:
            logger.error(f"Error fetching recently played: {e}")
            return pd.DataFrame()
    
    def get_audio_features(self, track_ids: List[str]) -> pd.DataFrame:
        """
        Get audio features for a list of tracks.
        
        Args:
            track_ids: List of Spotify track IDs
        """
        try:
            # Spotify API allows max 100 tracks per request
            all_features = []
            
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                features = self.sp.audio_features(batch)
                all_features.extend([f for f in features if f is not None])
                
                # Be nice to the API
                time.sleep(0.1)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(all_features)
            
            # Select relevant features for recommendation
            feature_columns = [
                'id', 'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'time_signature'
            ]
            
            if not features_df.empty:
                features_df = features_df[feature_columns]
                features_df.rename(columns={'id': 'track_id'}, inplace=True)
            
            logger.info(f"Retrieved audio features for {len(features_df)} tracks")
            return features_df
            
        except Exception as e:
            logger.error(f"Error fetching audio features: {e}")
            return pd.DataFrame()
    
    def get_user_playlists(self, limit: int = 50) -> pd.DataFrame:
        """Get user's playlists."""
        try:
            results = self.sp.current_user_playlists(limit=limit)
            playlists_data = []
            
            for item in results['items']:
                playlist_info = {
                    'playlist_id': item['id'],
                    'playlist_name': item['name'],
                    'track_count': item['tracks']['total'],
                    'public': item['public'],
                    'collaborative': item['collaborative']
                }
                playlists_data.append(playlist_info)
            
            logger.info(f"Retrieved {len(playlists_data)} playlists")
            return pd.DataFrame(playlists_data)
            
        except Exception as e:
            logger.error(f"Error fetching playlists: {e}")
            return pd.DataFrame()
    
    def get_playlist_tracks(self, playlist_id: str) -> pd.DataFrame:
        """Get tracks from a specific playlist."""
        try:
            results = self.sp.playlist_tracks(playlist_id)
            tracks_data = []
            
            for item in results['items']:
                if item['track'] and item['track']['id']:
                    track = item['track']
                    track_info = {
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'],
                        'album_name': track['album']['name'],
                        'added_at': item['added_at'],
                        'popularity': track['popularity']
                    }
                    tracks_data.append(track_info)
            
            logger.info(f"Retrieved {len(tracks_data)} tracks from playlist")
            return pd.DataFrame(tracks_data)
            
        except Exception as e:
            logger.error(f"Error fetching playlist tracks: {e}")
            return pd.DataFrame()
    
    def search_tracks(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Search for tracks."""
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            tracks_data = []
            
            for item in results['tracks']['items']:
                track_info = {
                    'track_id': item['id'],
                    'track_name': item['name'],
                    'artist_name': item['artists'][0]['name'],
                    'album_name': item['album']['name'],
                    'popularity': item['popularity']
                }
                tracks_data.append(track_info)
            
            return pd.DataFrame(tracks_data)
            
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return pd.DataFrame()
    
    def get_comprehensive_user_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive user data for recommendation system.
        
        Returns:
            Dictionary containing different types of user data
        """
        logger.info("Starting comprehensive data collection...")
        
        data = {}
        
        # Get top tracks (different time ranges)
        data['top_tracks_short'] = self.get_user_top_tracks(limit=50, time_range='short_term')
        data['top_tracks_medium'] = self.get_user_top_tracks(limit=50, time_range='medium_term')
        data['top_tracks_long'] = self.get_user_top_tracks(limit=50, time_range='long_term')
        
        # Get recently played
        data['recently_played'] = self.get_recently_played(limit=50)
        
        # Combine all track IDs for audio features
        all_track_ids = set()
        for df in data.values():
            if not df.empty and 'track_id' in df.columns:
                all_track_ids.update(df['track_id'].tolist())
        
        # Get audio features for all tracks
        if all_track_ids:
            data['audio_features'] = self.get_audio_features(list(all_track_ids))
        
        # Get playlists
        data['playlists'] = self.get_user_playlists()
        
        logger.info("Data collection completed")
        return data