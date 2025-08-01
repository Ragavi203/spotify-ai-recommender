import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class RecommendationVisualizer:
    """
    Visualization tools for recommendation system analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        
    def plot_user_preferences(self, preferences: Dict[str, float], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot user's musical preferences.
        
        Args:
            preferences: Dictionary of user preferences
            save_path: Optional path to save the plot
        """
        # Filter numeric preferences for audio features
        audio_features = {}
        for key, value in preferences.items():
            if key.startswith('avg_') and isinstance(value, (int, float)):
                feature_name = key.replace('avg_', '').title()
                audio_features[feature_name] = value
        
        if not audio_features:
            logger.warning("No audio feature preferences found")
            return
        
        # Create radar chart
        fig = go.Figure()
        
        features = list(audio_features.keys())
        values = list(audio_features.values())
        
        # Normalize values to 0-1 range for better visualization
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized_values = [0.5] * len(values)
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=features,
            fill='toself',
            name='User Preferences',
            line=dict(color='blue')
        ))