import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Spotify API Configuration
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8080')

# Model Configuration
DEFAULT_N_FACTORS = 50
DEFAULT_N_RECOMMENDATIONS = 20
RANDOM_STATE = 42

# Feature Configuration
AUDIO_FEATURES = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
]

# Paths
RESULTS_DIR = 'results'
VISUALIZATIONS_DIR = 'visualizations'
DATA_DIR = 'data'

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
