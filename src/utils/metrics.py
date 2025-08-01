import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)

class RecommendationMetrics:
    """
    Comprehensive metrics for evaluating recommendation systems.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def precision_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Precision@K - fraction of recommended items that are relevant.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant/liked item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k <= 0 or not recommended:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        return hits / min(k, len(recommended_k))
    
    def recall_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Recall@K - fraction of relevant items that are recommended.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant/liked item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if not relevant or not recommended:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        return hits / len(relevant_set)
    
    def f1_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate F1@K score.
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant/liked item IDs
            k: Number of top recommendations to consider
            
        Returns:
            F1@K score
        """
        precision = self.precision_at_k(recommended, relevant, k)
        recall = self.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, recommended: List[str], relevant: List[str]) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            recommended: List of recommended item IDs (in order)
            relevant: List of relevant/liked item IDs
            
        Returns:
            Average Precision score
        """
        if not relevant or not recommended:
            return 0.0
        
        relevant_set = set(relevant)
        precision_sum = 0.0
        hits = 0
        
        for i, item in enumerate(recommended):
            if item in relevant_set:
                hits += 1
                precision_sum += hits / (i + 1)
        
        return precision_sum / len(relevant_set) if relevant_set else 0.0
    
    def mean_average_precision(self, all_recommended: List[List[str]], 
                             all_relevant: List[List[str]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple users.
        
        Args:
            all_recommended: List of recommendation lists for each user
            all_relevant: List of relevant item lists for each user
            
        Returns:
            Mean Average Precision score
        """
        if len(all_recommended) != len(all_relevant):
            raise ValueError("Number of recommendation and relevant lists must match")
        
        if not all_recommended:
            return 0.0
        
        ap_scores = []
        for recommended, relevant in zip(all_recommended, all_relevant):
            ap = self.average_precision(recommended, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    def normalized_discounted_cumulative_gain(self, recommended: List[str], 
                                            relevant: List[str], 
                                            relevance_scores: Optional[Dict[str, float]] = None,
                                            k: Optional[int] = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            recommended: List of recommended item IDs
            relevant: List of relevant item IDs
            relevance_scores: Optional dict mapping item IDs to relevance scores
            k: Number of top recommendations to consider (None for all)
            
        Returns:
            NDCG@K score
        """
        if not recommended or not relevant:
            return 0.0
        
        if k is not None:
            recommended = recommended[:k]
        
        # Default relevance scores (binary)
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended):
            if item in relevance_scores:
                relevance = relevance_scores[item]
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        sorted_relevant = sorted(relevant, 
                               key=lambda x: relevance_scores.get(x, 0), 
                               reverse=True)
        
        if k is not None:
            sorted_relevant = sorted_relevant[:k]
        
        idcg = 0.0
        for i, item in enumerate(sorted_relevant):
            relevance = relevance_scores.get(item, 0)
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def diversity_score(self, recommended_items: List[str], 
                       similarity_matrix: pd.DataFrame) -> float:
        """
        Calculate diversity score based on pairwise similarities.
        
        Args:
            recommended_items: List of recommended item IDs
            similarity_matrix: DataFrame with pairwise item similarities
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(recommended_items) < 2:
            return 0.0
        
        # Get items that exist in similarity matrix
        valid_items = [item for item in recommended_items 
                      if item in similarity_matrix.index]
        
        if len(valid_items) < 2:
            return 0.0
        
        # Calculate average pairwise similarity
        total_similarity = 0.0
        pair_count = 0
        
        for i in range(len(valid_items)):
            for j in range(i + 1, len(valid_items)):
                item1, item2 = valid_items[i], valid_items[j]
                similarity = similarity_matrix.loc[item1, item2]
                total_similarity += similarity
                pair_count += 1
        
        avg_similarity = total_similarity / pair_count if pair_count > 0 else 0.0
        
        # Diversity is 1 - average similarity
        return 1.0 - avg_similarity
    
    def novelty_score(self, recommended_items: List[str], 
                     popularity_scores: Dict[str, float]) -> float:
        """
        Calculate novelty score based on item popularity.
        
        Args:
            recommended_items: List of recommended item IDs
            popularity_scores: Dict mapping item IDs to popularity scores
            
        Returns:
            Novelty score (higher = more novel/less popular items)
        """
        if not recommended_items:
            return 0.0
        
        # Get popularity scores for recommended items
        popularities = []
        for item in recommended_items:
            if item in popularity_scores:
                popularities.append(popularity_scores[item])
        
        if not popularities:
            return 0.0
        
        # Novelty is inverse of average popularity
        avg_popularity = np.mean(popularities)
        max_popularity = max(popularity_scores.values()) if popularity_scores else 1.0
        
        return 1.0 - (avg_popularity / max_popularity)
    
    def coverage_score(self, all_recommended: List[List[str]], 
                      all_items: List[str]) -> float:
        """
        Calculate catalog coverage - fraction of all items that get recommended.
        
        Args:
            all_recommended: List of recommendation lists
            all_items: List of all available items
            
        Returns:
            Coverage score (fraction of catalog covered)
        """
        if not all_items:
            return 0.0
        
        # Get all unique recommended items
        recommended_set = set()
        for rec_list in all_recommended:
            recommended_set.update(rec_list)
        
        return len(recommended_set) / len(all_items)
    
    def evaluate_recommendations(self, 
                               recommended: List[str],
                               relevant: List[str],
                               similarity_matrix: Optional[pd.DataFrame] = None,
                               popularity_scores: Optional[Dict[str, float]] = None,
                               k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Comprehensive evaluation of recommendations.
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant/liked items
            similarity_matrix: Optional similarity matrix for diversity
            popularity_scores: Optional popularity scores for novelty
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with all evaluation metrics
        """
        metrics = {}
        
        # Precision, Recall, F1 at different k values
        for k in k_values:
            metrics[f'precision_at_{k}'] = self.precision_at_k(recommended, relevant, k)
            metrics[f'recall_at_{k}'] = self.recall_at_k(recommended, relevant, k)
            metrics[f'f1_at_{k}'] = self.f1_at_k(recommended, relevant, k)
            metrics[f'ndcg_at_{k}'] = self.normalized_discounted_cumulative_gain(
                recommended, relevant, k=k
            )
        
        # Average Precision
        metrics['average_precision'] = self.average_precision(recommended, relevant)
        
        # Diversity
        if similarity_matrix is not None:
            metrics['diversity'] = self.diversity_score(recommended, similarity_matrix)
        
        # Novelty
        if popularity_scores is not None:
            metrics['novelty'] = self.novelty_score(recommended, popularity_scores)
        
        return metrics
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple recommendation models.
        
        Args:
            model_results: Dict with model names as keys and metrics dicts as values
            
        Returns:
            DataFrame comparing models across metrics
        """
        comparison_df = pd.DataFrame(model_results).T
        
        # Add ranking for each metric (1 = best)
        for column in comparison_df.columns:
            # For most metrics, higher is better
            comparison_df[f'{column}_rank'] = comparison_df[column].rank(ascending=False)
        
        return comparison_df
    
    def statistical_significance_test(self, 
                                    model1_scores: List[float],
                                    model2_scores: List[float],
                                    test_type: str = 'paired_t') -> Dict[str, float]:
        """
        Test statistical significance between two models.
        
        Args:
            model1_scores: Evaluation scores for model 1
            model2_scores: Evaluation scores for model 2
            test_type: Type of test ('paired_t', 'wilcoxon')
            
        Returns:
            Dictionary with test results
        """
        from scipy import stats
        
        if len(model1_scores) != len(model2_scores):
            raise ValueError("Score lists must have same length")
        
        results = {
            'model1_mean': np.mean(model1_scores),
            'model2_mean': np.mean(model2_scores),
            'difference': np.mean(model1_scores) - np.mean(model2_scores)
        }
        
        if test_type == 'paired_t':
            statistic, p_value = stats.ttest_rel(model1_scores, model2_scores)
            results['test'] = 'paired_t_test'
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(model1_scores, model2_scores)
            results['test'] = 'wilcoxon_signed_rank'
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        results['statistic'] = statistic
        results['p_value'] = p_value
        results['significant'] = p_value < 0.05
        
        return results