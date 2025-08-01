# Spotify AI Recommendation System

A machine learning-powered music recommendation system that analyzes Spotify listening history to generate personalized music recommendations using collaborative filtering and content-based algorithms.

## Overview

This system integrates with the Spotify Web API to collect user listening data, performs comprehensive music preference analysis, and generates intelligent recommendations using multiple machine learning approaches. The system implements robust error handling and fallback strategies to ensure reliable performance across different API availability scenarios.

## Technical Architecture

### Core Components

- **Data Collection Layer**: Spotify Web API integration with OAuth 2.0 authentication
- **Processing Engine**: Audio feature analysis and user preference profiling
- **Recommendation Engine**: Multi-algorithm approach combining collaborative and content-based filtering
- **Evaluation Framework**: Confidence scoring and recommendation quality metrics

### Machine Learning Approaches

1. **Artist-Based Collaborative Filtering**: Analyzes user listening patterns to identify preferred artists and recommends tracks from similar artists
2. **Content-Based Filtering**: Utilizes Spotify's audio features (danceability, energy, valence, acousticness) for track similarity matching
3. **Hybrid Recommendation**: Combines multiple approaches with weighted scoring for optimal results

## Features

- Comprehensive Spotify listening history analysis
- Multi-timeframe preference analysis (short, medium, long-term)
- Real-time recommendation generation
- Robust API error handling with intelligent fallback strategies
- Confidence scoring for recommendation quality assessment
- Support for diverse music libraries and genres

## Technology Stack

- **Python 3.8+**: Core implementation language
- **Spotipy**: Spotify Web API Python wrapper
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Python-dotenv**: Environment variable management

## Installation

### Prerequisites

- Python 3.8 or higher
- Spotify account (Premium recommended for full API access)
- Spotify Developer Application credentials

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Ragavi203/spotify-ai-recommender.git
cd spotify-ai-recommender
```

2. Create and activate virtual environment:
```bash
python3 -m venv spotify-env
source spotify-env/bin/activate  # Windows: spotify-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` file with your Spotify application credentials:
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8080
```

## Spotify Developer Setup

1. Navigate to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Create new application
3. Configure redirect URI: `http://127.0.0.1:8080`
4. Note Client ID and Client Secret for environment configuration

## Usage

### Basic Execution

```bash
python src/main.py
```

### Expected Output

The system generates comprehensive analysis including:

- User music profile with popularity metrics
- Top artist analysis with listening frequency
- Recent favorites identification
- Personalized recommendations with confidence scores

### Example Output Format

```
Spotify AI Recommendation System
Loading environment variables...
CLIENT_ID: 21ee24a6...
CLIENT_SECRET: aae82dfa...

Music Analysis Results
Average Track Popularity: 68.5/100

TOP ARTISTS:
1. Lana Del Rey (7 tracks)
2. A.R. Rahman (3 tracks)
3. Anirudh Ravichander (2 tracks)

RECOMMENDED FOR YOU:
1. Young And Beautiful by Lana Del Rey (Match: 0.36)
2. Summertime Sadness by Lana Del Rey (Match: 0.36)
3. Jinguchaa by A.R. Rahman (Match: 0.26)
```

## Project Structure

```
spotify-ai-recommender/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── spotify_client.py          # Spotify API integration
│   │   └── data_processor.py          # Data preprocessing utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py # Collaborative filtering algorithms
│   │   └── content_based.py          # Content-based filtering
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── visualizations.py         # Data visualization utilities
│   └── main.py                       # Main application entry point
├── requirements.txt                   # Python dependencies
├── config.py                         # Configuration management
├── .env.example                      # Environment template
├── .gitignore                        # Git ignore rules
└── README.md                         # Documentation
```

## Algorithm Implementation

### Data Collection Process

1. **Authentication**: OAuth 2.0 flow with required scopes
2. **Multi-endpoint Data Retrieval**: 
   - User top tracks (short/medium/long term)
   - Recently played tracks
   - Audio features extraction
3. **Data Validation**: Comprehensive error handling and data quality checks

### Recommendation Generation

#### Artist-Based Algorithm
- Analyzes listening frequency across different time periods
- Identifies user preference patterns for specific artists
- Generates recommendations from preferred artists' catalogs
- Applies popularity and recency weighting

#### Content-Based Algorithm
- Extracts audio features (danceability, energy, valence, acousticness, speechiness, tempo)
- Calculates user preference profiles based on listening history
- Implements similarity matching using feature vector analysis
- Applies confidence scoring based on feature alignment

### Scoring Methodology

- **Artist Preference Score**: Calculated from listening frequency and time-weighted analysis
- **Popularity Score**: Incorporates track popularity metrics
- **Similarity Score**: Audio feature matching confidence
- **Final Score**: Weighted combination of multiple scoring factors

## Error Handling

The system implements comprehensive error handling:

- **API Rate Limiting**: Automatic retry with exponential backoff
- **Authentication Failures**: Clear error messaging and resolution guidance
- **Data Availability**: Graceful fallback from audio features to artist-based recommendations
- **Network Issues**: Robust connection management and timeout handling

## Performance Characteristics

- **Data Processing**: Efficiently handles 100+ tracks with optimized batch processing
- **API Integration**: Implements rate limiting and request optimization
- **Memory Usage**: Optimized data structures for large music libraries
- **Response Time**: Sub-second analysis for typical user libraries

## Configuration Options

### Environment Variables
- `SPOTIFY_CLIENT_ID`: Spotify application client identifier
- `SPOTIFY_CLIENT_SECRET`: Spotify application secret key
- `SPOTIFY_REDIRECT_URI`: OAuth redirect endpoint

### Customizable Parameters
- Recommendation count limits
- Time period weights for preference analysis
- Audio feature importance weights
- Confidence threshold settings

## Troubleshooting

### Common Issues

**Authentication Errors (HTTP 403)**
- Verify Spotify credentials in `.env` file
- Ensure redirect URI matches Spotify app configuration
- Check application scope permissions

**No Data Retrieved**
- Confirm active Spotify listening history
- Verify account has sufficient listening data
- Check API endpoint availability

**Import Errors**
- Activate virtual environment
- Install all requirements: `pip install -r requirements.txt`
- Verify Python version compatibility

### API Limitations

- Some endpoints require Spotify Premium subscription
- Rate limiting may affect large-scale data collection
- Audio features availability depends on track catalog coverage

## Testing

### Unit Testing
```bash
python -m pytest tests/
```

### Integration Testing
```bash
python tests/integration_test.py
```

## Contributing

### Development Setup
1. Fork repository
2. Create feature branch
3. Implement changes with appropriate testing
4. Submit pull request with detailed description

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Implement appropriate error handling
- Add unit tests for new functionality

## License

MIT License - see LICENSE file for details.

## Dependencies

### Core Requirements
- spotipy==2.23.0
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- python-dotenv==1.0.0

### Development Dependencies
- jupyter==1.0.0
- matplotlib==3.7.2
- seaborn==0.12.2
- plotly==5.15.0

## API Reference

### Spotify Web API Endpoints Used
- `/v1/me/top/tracks`: User top tracks retrieval
- `/v1/me/player/recently-played`: Recent listening history
- `/v1/audio-features`: Track audio feature analysis
- `/v1/me/playlists`: User playlist information

## Security Considerations

- Environment variables used for credential management
- No hardcoded API keys in source code
- OAuth 2.0 implementation following security best practices
- Proper scope limitation for API access

## Performance Optimization

- Batch processing for API requests
- Efficient data structures for large datasets
- Caching mechanisms for repeated operations
- Optimized algorithm implementations for scalability
