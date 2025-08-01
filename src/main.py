import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables first
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.data.spotify_client import SpotifyClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSpotifyRecommender:
    """
    Simplified Spotify recommendation system for initial testing.
    """
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize the recommendation system."""
        self.spotify_client = SpotifyClient(client_id, client_secret, redirect_uri)
        self.user_data = {}
        
    def collect_user_data(self) -> bool:
        """Collect user data from Spotify."""
        logger.info("Starting user data collection...")
        
        try:
            # Get comprehensive user data
            self.user_data = self.spotify_client.get_comprehensive_user_data()
            
            if not any(not df.empty for df in self.user_data.values() if isinstance(df, pd.DataFrame)):
                logger.error("No user data collected")
                return False
            
            # Log data summary
            total_tracks = 0
            for key, df in self.user_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info(f"✓ {key}: {len(df)} records")
                    if 'track_id' in df.columns:
                        total_tracks += len(df['track_id'].unique())
            
            logger.info(f"🎯 Total unique tracks collected: {total_tracks}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting user data: {e}")
            return False
    
    def analyze_user_music(self) -> Dict[str, any]:
        """Analyze user's music preferences."""
        analysis = {}
        
        try:
            # Analyze top tracks
            if 'top_tracks_short' in self.user_data and not self.user_data['top_tracks_short'].empty:
                top_tracks = self.user_data['top_tracks_short']
                
                # Top artists
                top_artists = top_tracks['artist_name'].value_counts().head(5)
                analysis['top_artists'] = top_artists.to_dict()
                
                # Average popularity
                analysis['avg_popularity'] = top_tracks['popularity'].mean()
                
                # Most recent favorites
                analysis['recent_favorites'] = [
                    f"{row['track_name']} by {row['artist_name']}"
                    for _, row in top_tracks.head(5).iterrows()
                ]
            
            # Analyze audio features if available
            if 'audio_features' in self.user_data and not self.user_data['audio_features'].empty:
                audio_features = self.user_data['audio_features']
                
                # Get tracks that are in top tracks
                if 'top_tracks_short' in self.user_data:
                    top_track_ids = self.user_data['top_tracks_short']['track_id'].tolist()
                    user_audio_features = audio_features[audio_features['track_id'].isin(top_track_ids)]
                    
                    if not user_audio_features.empty:
                        # Calculate average features
                        feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']
                        
                        for feature in feature_cols:
                            if feature in user_audio_features.columns:
                                analysis[f'avg_{feature}'] = float(user_audio_features[feature].mean())
                        
                        # Categorize music taste
                        energy = analysis.get('avg_energy', 0.5)
                        valence = analysis.get('avg_valence', 0.5)
                        danceability = analysis.get('avg_danceability', 0.5)
                        
                        if energy > 0.7 and valence > 0.7:
                            analysis['music_mood'] = "High Energy & Happy"
                        elif energy < 0.3 and valence < 0.3:
                            analysis['music_mood'] = "Low Energy & Melancholic"
                        elif energy > 0.6 and danceability > 0.6:
                            analysis['music_mood'] = "Dance & Party"
                        elif analysis.get('avg_acousticness', 0) > 0.6:
                            analysis['music_mood'] = "Acoustic & Chill"
                        else:
                            analysis['music_mood'] = "Balanced Mix"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing user music: {e}")
            return {}
    
    def get_simple_recommendations(self, n_recommendations: int = 20) -> List[Tuple[str, str, float]]:
        """Get simple content-based recommendations."""
        recommendations = []
        
        try:
            # If we don't have audio features, use artist-based recommendations
            if 'audio_features' not in self.user_data or self.user_data['audio_features'].empty:
                logger.info("Using artist-based recommendations (no audio features available)")
                return self._get_artist_based_recommendations(n_recommendations)
            
            # Audio feature-based recommendations (original logic)
            if 'top_tracks_short' not in self.user_data or self.user_data['top_tracks_short'].empty:
                logger.warning("No top tracks available for recommendations")
                return recommendations
            
            # Get user's favorite tracks and their features
            top_tracks = self.user_data['top_tracks_short']
            audio_features = self.user_data['audio_features']
            
            # Get audio features for top tracks
            top_track_ids = top_tracks['track_id'].tolist()
            user_features = audio_features[audio_features['track_id'].isin(top_track_ids)]
            
            if user_features.empty:
                logger.warning("No audio features found for top tracks, falling back to artist-based")
                return self._get_artist_based_recommendations(n_recommendations)
            
            # Calculate user's average preferences
            feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'tempo']
            available_features = [col for col in feature_cols if col in user_features.columns]
            
            if not available_features:
                logger.warning("No suitable audio features found, falling back to artist-based")
                return self._get_artist_based_recommendations(n_recommendations)
            
            user_profile = user_features[available_features].mean()
            
            # Find similar tracks from all available tracks
            all_tracks = audio_features.copy()
            
            # Remove tracks user already knows
            known_track_ids = set()
            for key, df in self.user_data.items():
                if isinstance(df, pd.DataFrame) and 'track_id' in df.columns:
                    known_track_ids.update(df['track_id'].tolist())
            
            candidate_tracks = all_tracks[~all_tracks['track_id'].isin(known_track_ids)]
            
            if candidate_tracks.empty:
                logger.warning("No candidate tracks found for recommendations")
                return self._get_artist_based_recommendations(n_recommendations)
            
            # Calculate similarity scores
            similarities = []
            
            for _, track in candidate_tracks.iterrows():
                # Calculate cosine similarity
                track_features = track[available_features]
                
                # Simple similarity calculation (could be improved with proper cosine similarity)
                similarity = 0
                for feature in available_features:
                    diff = abs(user_profile[feature] - track_features[feature])
                    similarity += (1 - diff)  # Higher similarity for smaller differences
                
                similarity = similarity / len(available_features)  # Normalize
                similarities.append((track['track_id'], similarity))
            
            # Sort by similarity and get top recommendations
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:n_recommendations]
            
            # Get track names for recommendations
            for track_id, similarity in top_similarities:
                # Try to find track name from collected data
                track_name = "Unknown Track"
                
                # Search in all collected data
                for key, df in self.user_data.items():
                    if isinstance(df, pd.DataFrame) and 'track_id' in df.columns:
                        match = df[df['track_id'] == track_id]
                        if not match.empty and 'track_name' in match.columns:
                            track_name = f"{match.iloc[0]['track_name']} by {match.iloc[0].get('artist_name', 'Unknown Artist')}"
                            break
                
                recommendations.append((track_id, track_name, similarity))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_artist_based_recommendations(n_recommendations)
    
    def _get_artist_based_recommendations(self, n_recommendations: int = 20) -> List[Tuple[str, str, float]]:
        """Get recommendations based on favorite artists when audio features aren't available."""
        recommendations = []
        
        try:
            # Get user's top artists
            if 'top_tracks_short' not in self.user_data or self.user_data['top_tracks_short'].empty:
                return recommendations
            
            top_tracks = self.user_data['top_tracks_short']
            
            # Get artist preferences
            artist_counts = top_tracks['artist_name'].value_counts()
            favorite_artists = set(artist_counts.head(10).index)
            
            # Find tracks from favorite artists in other time periods
            candidate_tracks = []
            
            # Look in medium and long term tracks
            for time_period in ['top_tracks_medium', 'top_tracks_long']:
                if time_period in self.user_data and not self.user_data[time_period].empty:
                    period_tracks = self.user_data[time_period]
                    
                    # Find tracks by favorite artists that user hasn't heard recently
                    known_recent_tracks = set(top_tracks['track_id'])
                    
                    for _, track in period_tracks.iterrows():
                        if (track['artist_name'] in favorite_artists and 
                            track['track_id'] not in known_recent_tracks):
                            
                            # Score based on artist popularity in user's listening
                            artist_score = artist_counts.get(track['artist_name'], 0) / len(top_tracks)
                            popularity_score = track.get('popularity', 50) / 100
                            
                            # Combined score
                            final_score = (artist_score * 0.7) + (popularity_score * 0.3)
                            
                            track_name = f"{track['track_name']} by {track['artist_name']}"
                            candidate_tracks.append((track['track_id'], track_name, final_score))
            
            # Sort by score and return top recommendations
            candidate_tracks.sort(key=lambda x: x[2], reverse=True)
            recommendations = candidate_tracks[:n_recommendations]
            
            if recommendations:
                logger.info(f"Generated {len(recommendations)} artist-based recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating artist-based recommendations: {e}")
            return []
    
    def run_analysis(self) -> Dict[str, any]:
        """Run complete analysis and recommendation pipeline."""
        logger.info("Starting Spotify music analysis...")
        
        results = {
            'success': False,
            'user_analysis': {},
            'recommendations': []
        }
        
        try:
            # Step 1: Collect data
            if not self.collect_user_data():
                logger.error("Data collection failed")
                return results
            
            # Step 2: Analyze user preferences
            user_analysis = self.analyze_user_music()
            if user_analysis:
                logger.info("✓ User music analysis completed")
                results['user_analysis'] = user_analysis
            
            # Step 3: Generate simple recommendations
            recommendations = self.get_simple_recommendations()
            if recommendations:
                logger.info(f"✓ Generated {len(recommendations)} recommendations")
                results['recommendations'] = recommendations
            
            results['success'] = True
            logger.info("🎉 Analysis completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            results['error'] = str(e)
            return results


def main():
    """Main function to run the recommendation system."""
    print("🎵 Spotify AI Recommendation System")
    print("=" * 40)
    
    # Debug environment loading
    print("Loading environment variables...")
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8080')
    
    print(f"CLIENT_ID: {client_id[:8] + '...' if client_id else 'NOT SET'}")
    print(f"CLIENT_SECRET: {client_secret[:8] + '...' if client_secret else 'NOT SET'}")
    print(f"REDIRECT_URI: {redirect_uri}")
    
    if not client_id or not client_secret:
        print("Please set your Spotify credentials in the .env file")
        print("\nTo fix this:")
        print("1. Edit the .env file in your project root")
        print("2. Add your Spotify app credentials:")
        print("   SPOTIFY_CLIENT_ID=your_client_id")
        print("   SPOTIFY_CLIENT_SECRET=your_client_secret")
        print("3. Get credentials from: https://developer.spotify.com/dashboard/")
        return
    
    try:
        # Initialize recommendation system
        print("\n🔗 Initializing Spotify client...")
        recommender = SimpleSpotifyRecommender(client_id, client_secret, redirect_uri)
        
        # Run analysis
        results = recommender.run_analysis()
        
        if results['success']:
            print("\n🎵 Your Music Analysis Results 🎵")
            print("=" * 50)
            
            # Display user analysis
            if results['user_analysis']:
                analysis = results['user_analysis']
                
                print(f"\nMUSIC PROFILE:")
                print("-" * 20)
                
                if 'music_mood' in analysis:
                    print(f" Music Mood: {analysis['music_mood']}")
                
                if 'avg_popularity' in analysis:
                    print(f" Average Track Popularity: {analysis['avg_popularity']:.1f}/100")
                
                # Display audio features
                audio_features = ['energy', 'valence', 'danceability', 'acousticness']
                for feature in audio_features:
                    key = f'avg_{feature}'
                    if key in analysis:
                        print(f" {feature.title()}: {analysis[key]:.2f}")
                
                # Display top artists
                if 'top_artists' in analysis:
                    print(f"\n TOP ARTISTS:")
                    for i, (artist, count) in enumerate(analysis['top_artists'].items(), 1):
                        print(f"  {i}. {artist} ({count} tracks)")
                
                # Display recent favorites
                if 'recent_favorites' in analysis:
                    print(f"\n RECENT FAVORITES:")
                    for i, track in enumerate(analysis['recent_favorites'], 1):
                        print(f"  {i}. {track}")
            
            # Display recommendations
            if results['recommendations']:
                print(f"\n RECOMMENDED FOR YOU:")
                print("-" * 30)
                
                for i, (track_id, track_name, similarity) in enumerate(results['recommendations'][:10], 1):
                    print(f"{i:2d}. {track_name} (Match: {similarity:.2f})")
            
            print(f"\n Analysis complete! 🎉")
            
        else:
            print(" Analysis failed. Check the error messages above.")
            if 'error' in results:
                print(f"Error: {results['error']}")
    
    except Exception as e:
        print(f" Unexpected error: {e}")
        print(" Make sure:")
        print("  - Your Spotify credentials are correct")
        print("  - You've authorized the app in your browser")
        print("  - You have an active internet connection")
        print("  - You've been listening to music on Spotify recently")


if __name__ == "__main__":
    main()
