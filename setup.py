#!/usr/bin/env python3
"""
Setup script for Spotify AI Recommendation System
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    dirs = [
        'src',
        'src/data',
        'src/models', 
        'src/utils',
        'notebooks',
        'results',
        'visualizations',
        'data'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if dir_path.startswith('src/'):
            init_file = Path(dir_path) / '__init__.py'
            init_file.touch()
    
    print("✓ Directory structure created")

def create_env_file():
    """Create a .env template file."""
    env_content = """# Spotify API Credentials
# Get these from https://developer.spotify.com/dashboard/
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8080

# Optional: Set logging level
LOG_LEVEL=INFO
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✓ .env template created - please add your Spotify credentials")

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def create_config_file():
    """Create a config.py file."""
    config_content = """import os
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
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("✓ Configuration file created")

def create_example_notebook():
    """Create an example Jupyter notebook."""
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Spotify AI Recommendation System\\n",
    "\\n",
    "This notebook demonstrates how to use the Spotify recommendation system."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "sys.path.append('../src')\\n",
    "\\n",
    "from main import SpotifyRecommendationSystem\\n",
    "import config"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the recommendation system\\n",
    "recommender = SpotifyRecommendationSystem(\\n",
    "    client_id=config.SPOTIFY_CLIENT_ID,\\n",
    "    client_secret=config.SPOTIFY_CLIENT_SECRET,\\n",
    "    redirect_uri=config.SPOTIFY_REDIRECT_URI\\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run the complete pipeline\\n",
    "results = recommender.run_complete_pipeline()\\n",
    "\\n",
    "if results['success']:\\n",
    "    print('✅ Recommendations generated successfully!')\\n",
    "    \\n",
    "    # Display top recommendations\\n",
    "    for model_name, recs in results['recommendations'].items():\\n",
    "        print(f'\\\\n{model_name.upper()} - Top 5:')\\n",
    "        for i, (track_id, score) in enumerate(recs[:5], 1):\\n",
    "            print(f'{i}. {track_id} (Score: {score:.3f})')\\n",
    "else:\\n",
    "    print('❌ Failed to generate recommendations')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    with open('notebooks/spotify_recommendations_demo.ipynb', 'w') as f:
        f.write(notebook_content)
    
    print("✓ Example notebook created")

def create_readme():
    """Create README.md file."""
    readme_content = """# Spotify AI Recommendation System

An advanced music recommendation system that combines collaborative filtering, content-based filtering, and neural networks to provide personalized music recommendations.

## Features

- **Multiple Recommendation Algorithms**:
  - Collaborative Filtering (SVD, Surprise library)
  - Content-Based Filtering (Audio features)
  - Implicit Collaborative Filtering
  - Hybrid Recommendations

- **Comprehensive Evaluation**:
  - Precision@K, Recall@K, F1@K
  - NDCG (Normalized Discounted Cumulative Gain)
  - Diversity and Novelty metrics
  - Statistical significance testing

- **Rich Visualizations**:
  - User preference radar charts
  - Model performance comparisons
  - Audio feature analysis
  - Recommendation diversity plots

## Quick Start

1. **Setup the project**:
   ```bash
   python setup.py
   ```

2. **Get Spotify API credentials**:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
   - Create a new app
   - Copy Client ID and Client Secret to `.env` file

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the recommendation system**:
   ```bash
   python src/main.py
   ```

## Project Structure

```
spotify-ai-recommender/
├── src/
│   ├── data/
│   │   ├── spotify_client.py      # Spotify API wrapper
│   │   └── data_processor.py      # Data preprocessing
│   ├── models/
│   │   ├── collaborative_filtering.py
│   │   └── content_based.py
│   ├── utils/
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── visualizations.py     # Plotting utilities
│   └── main.py                    # Main application
├── notebooks/
│   └── spotify_recommendations_demo.ipynb
├── requirements.txt
├── config.py
└── README.md
```

## Usage Examples

### Basic Usage
```python
from src.main import SpotifyRecommendationSystem

# Initialize
recommender = SpotifyRecommendationSystem(
    client_id="your_client_id",
    client_secret="your_client_secret",
    redirect_uri="http://localhost:8080"
)

# Run complete pipeline
results = recommender.run_complete_pipeline()

# Get recommendations
recommendations = results['recommendations']
```

### Advanced Usage
```python
# Collect data
recommender.collect_user_data()

# Process data
recommender.process_data()

# Train models
recommender.train_models()

# Generate recommendations
recs = recommender.generate_recommendations(n_recommendations=50)

# Analyze results
analysis = recommender.analyze_recommendations()
```

## Configuration

Edit `config.py` to customize:
- Model parameters (number of factors, iterations)
- Audio features to use
- Output directories
- Logging settings

## Evaluation Metrics

The system provides comprehensive evaluation:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended  
- **F1@K**: Harmonic mean of precision and recall
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Diversity**: How different recommended items are from each other
- **Novelty**: How much the system recommends less popular items

## Visualization

Automatic generation of:
- User preference radar charts
- Model comparison plots
- Audio feature correlation matrices
- Recommendation diversity analysis
- Score distribution plots

## Requirements

- Python 3.8+
- Spotify Premium account (for full API access)
- Spotify Developer App credentials

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Troubleshooting

**Common Issues:**

1. **"No user data collected"**
   - Ensure your Spotify credentials are correct
   - Make sure you've played music recently
   - Check that you've authorized the app

2. **"Model training failed"**
   - Verify you have sufficient listening history
   - Check that audio features were retrieved successfully

3. **Import errors**
   - Run `pip install -r requirements.txt`
   - Ensure you're in the correct directory

For more help, check the example notebook in `notebooks/spotify_recommendations_demo.ipynb`.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✓ README.md created")

def main():
    """Main setup function."""
    print("🎵 Setting up Spotify AI Recommendation System...")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_env_file()
    create_config_file()
    
    
    # Ask about installing dependencies
    print("\n" + "=" * 50)
    install_deps = input("Would you like to install dependencies now? (y/N): ").lower().strip()
    
    if install_deps in ['y', 'yes']:
        if install_dependencies():
            print("\n🎉 Setup completed successfully!")
            print("\nNext steps:")
            print("1. Add your Spotify credentials to the .env file")
            print("2. Run: python src/main.py")
            print("3. Or explore the Jupyter notebook: notebooks/spotify_recommendations_demo.ipynb")
        else:
            print("\n⚠️ Setup completed with warnings. Please install dependencies manually:")
            print("pip install -r requirements.txt")
    else:
        print("\n✓ Setup completed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Add your Spotify credentials to the .env file")
        print("3. Run: python src/main.py")
    
    print("\n📚 For detailed instructions, see README.md")

if __name__ == "__main__":
    main()