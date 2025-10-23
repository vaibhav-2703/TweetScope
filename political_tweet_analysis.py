# Political Tweet Analysis: Graph-Based Link Prediction
# Clean Implementation

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PoliticalTweetProcessor:
    """Process political tweets for graph-based analysis"""
    
    def __init__(self):
        self.datasets = {}
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
    def load_dataset(self, filepath, text_column, sample_size=None):
        """Load and preprocess dataset"""
        try:
            df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
            
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            df[text_column] = df[text_column].astype(str)
            df = df[df[text_column].str.len() > 10]
            
            return df
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def preprocess_text(self, text):
        """Clean tweet text for analysis"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text.strip()

class CooccurrenceGraphBuilder:
    """Build co-occurrence graphs from political tweets"""
    
    def __init__(self, min_word_freq=3, window_size=5):
        self.min_word_freq = min_word_freq
        self.window_size = window_size
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
    def tokenize_text(self, text):
        """Tokenize and clean text"""
        words = text.lower().split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        return words
    
    def build_graph(self, texts):
        """Build co-occurrence graph from texts"""
        print(f"Processing {len(texts)} texts...")
        
        word_counts = Counter()
        tokenized_texts = []
        
        for text in texts:
            words = self.tokenize_text(text)
            tokenized_texts.append(words)
            for word in words:
                word_counts[word] += 1
        
        vocabulary = {word for word, count in word_counts.items() 
                    if count >= self.min_word_freq}
        
        print(f"Vocabulary size: {len(vocabulary)} words")
        
        G = nx.Graph()
        edge_weights = defaultdict(int)
        
        for words in tokenized_texts:
            filtered_words = [w for w in words if w in vocabulary]
            
            for i in range(len(filtered_words)):
                for j in range(i + 1, min(i + self.window_size, len(filtered_words))):
                    word1, word2 = filtered_words[i], filtered_words[j]
                    if word1 != word2:
                        edge_weights[(word1, word2)] += 1
        
        for (word1, word2), weight in edge_weights.items():
            G.add_edge(word1, word2, weight=weight)
        
        print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

class LinkPredictionEvaluator:
    """Evaluate different link prediction methods"""
    
    def __init__(self, test_ratio=0.2):
        self.test_ratio = test_ratio
        
    def common_neighbors_score(self, G, u, v):
        """Common neighbors link prediction"""
        if not G.has_node(u) or not G.has_node(v):
            return 0
        return len(list(nx.common_neighbors(G, u, v)))
    
    def jaccard_coefficient(self, G, u, v):
        """Jaccard coefficient"""
        if not G.has_node(u) or not G.has_node(v):
            return 0
        try:
            return next(nx.jaccard_coefficient(G, [(u, v)]), (None, None, 0))[2]
        except:
            return 0
    
    def adamic_adar_index(self, G, u, v):
        """Adamic-Adar index"""
        if not G.has_node(u) or not G.has_node(v):
            return 0
        try:
            return next(nx.adamic_adar_index(G, [(u, v)]), (None, None, 0))[2]
        except:
            return 0
    
    def preferential_attachment(self, G, u, v):
        """Preferential attachment"""
        if not G.has_node(u) or not G.has_node(v):
            return 0
        try:
            return next(nx.preferential_attachment(G, [(u, v)]), (None, None, 0))[2]
        except:
            return 0
    
    def generate_negative_samples(self, G, num_samples):
        """Generate negative samples (non-existent edges)"""
        import random
        nodes = list(G.nodes())
        negative_edges = set()
        
        while len(negative_edges) < num_samples:
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                negative_edges.add((u, v))
        
        return list(negative_edges)
    
    def evaluate_method(self, G, method_func):
        """Evaluate a single link prediction method"""
        edges = list(G.edges())
        
        if len(edges) < 10:
            return {'AUC': 0.0, 'AP': 0.0}
        
        train_edges, test_edges = train_test_split(
            edges, test_size=self.test_ratio, random_state=42
        )
        
        G_train = G.copy()
        G_train.remove_edges_from(test_edges)
        
        neg_edges = self.generate_negative_samples(G_train, len(test_edges))
        
        pos_scores = [method_func(G_train, u, v) for u, v in test_edges]
        neg_scores = [method_func(G_train, u, v) for u, v in neg_edges]
        
        y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
        y_scores = pos_scores + neg_scores
        
        try:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
        except:
            auc, ap = 0.0, 0.0
        
        return {'AUC': auc, 'AP': ap}
    
    def evaluate_all_methods(self, G):
        """Evaluate all link prediction methods"""
        methods = {
            'Common Neighbors': self.common_neighbors_score,
            'Jaccard Coefficient': self.jaccard_coefficient,
            'Adamic-Adar': self.adamic_adar_index,
            'Preferential Attachment': self.preferential_attachment
        }
        
        results = {}
        for method_name, method_func in methods.items():
            results[method_name] = self.evaluate_method(G, method_func)
        
        return results

def main():
    """Main analysis pipeline"""
    print("Political Tweet Analysis: Graph-Based Link Prediction")
    print("=" * 60)
    
    # Load datasets
    processor = PoliticalTweetProcessor()
    
    dataset_configs = {
        'us_elections': {
            'path': '/content/hashtag_donaldtrump.csv',
            'text_col': 'tweet',
            'sample_size': 15000
        },
        'indian_politics': {
            'path': '/content/tweets.csv', 
            'text_col': 'Tweet',
            'sample_size': 15000
        },
        'european_elections': {
            'path': '/content/EuropeanElection2019_EN.csv',
            'text_col': 'text',
            'sample_size': 15000
        }
    }
    
    print("Loading political tweet datasets...")
    for name, config in dataset_configs.items():
        df = processor.load_dataset(config['path'], config['text_col'], config['sample_size'])
        if df is not None:
            processor.datasets[name] = df
            print(f"✓ {name}: {len(df)} tweets loaded")
        else:
            print(f"✗ Failed to load {name}")
    
    print(f"\nTotal datasets loaded: {len(processor.datasets)}")
    print(f"Total tweets: {sum(len(df) for df in processor.datasets.values())}")
    
    # Build graphs
    print("\nBuilding co-occurrence graphs...")
    print("=" * 50)
    
    graph_builder = CooccurrenceGraphBuilder(min_word_freq=3, window_size=5)
    graphs = {}
    
    for dataset_name, df in processor.datasets.items():
        print(f"\nProcessing {dataset_name}...")
        
        text_col = 'tweet' if 'tweet' in df.columns else 'Tweet' if 'Tweet' in df.columns else 'text'
        texts = df[text_col].tolist()
        
        G = graph_builder.build_graph(texts)
        graphs[dataset_name] = G
        
        with open(f'/content/{dataset_name}_graph.pkl', 'wb') as f:
            pickle.dump(G, f)
        
        print(f"✓ Graph saved: {dataset_name}_graph.pkl")
    
    # Evaluate link prediction methods
    print("\nEvaluating link prediction methods...")
    print("=" * 50)
    
    evaluator = LinkPredictionEvaluator()
    all_results = {}
    
    for dataset_name, G in graphs.items():
        print(f"\nEvaluating {dataset_name} dataset...")
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        results = evaluator.evaluate_all_methods(G)
        all_results[dataset_name] = results
        
        for method, metrics in results.items():
            print(f"{method:20s}: AUC={metrics['AUC']:.4f}, AP={metrics['AP']:.4f}")
    
    # Results analysis
    print("\nResults Analysis")
    print("=" * 30)
    
    results_data = []
    for dataset, methods in all_results.items():
        for method, metrics in methods.items():
            results_data.append({
                'Dataset': dataset.replace('_', ' ').title(),
                'Method': method,
                'AUC': metrics['AUC'],
                'AP': metrics['AP']
            })
    
    df_results = pd.DataFrame(results_data)
    avg_performance = df_results.groupby('Method')[['AUC', 'AP']].mean().sort_values('AUC', ascending=False)
    
    print("\nAverage Performance Across Datasets:")
    print("-" * 40)
    for method, metrics in avg_performance.iterrows():
        print(f"{method:20s}: AUC={metrics['AUC']:.4f}, AP={metrics['AP']:.4f}")
    
    best_method = avg_performance.index[0]
    best_auc = avg_performance['AUC'].iloc[0]
    
    print(f"\nBest performing method: {best_method}")
    print(f"Best AUC score: {best_auc:.4f}")
    
    if best_auc >= 0.9:
        performance_level = "Excellent"
    elif best_auc >= 0.8:
        performance_level = "Good"
    elif best_auc >= 0.7:
        performance_level = "Fair"
    else:
        performance_level = "Poor"
    
    print(f"Overall performance level: {performance_level}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.barplot(data=df_results, x='Method', y='AUC', hue='Dataset', ax=axes[0,0])
    axes[0,0].set_title('AUC Comparison Across Methods')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df_results, x='Method', y='AP', hue='Dataset', ax=axes[0,1])
    axes[0,1].set_title('Average Precision Comparison')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    pivot_auc = df_results.pivot(index='Dataset', columns='Method', values='AUC')
    sns.heatmap(pivot_auc, annot=True, cmap='YlOrRd', fmt='.3f', ax=axes[1,0])
    axes[1,0].set_title('AUC Heatmap')
    
    method_ranking = avg_performance.reset_index()
    sns.barplot(data=method_ranking, x='AUC', y='Method', ax=axes[1,1])
    axes[1,1].set_title('Method Ranking by Average AUC')
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    results_summary = {
        'best_method': best_method,
        'best_auc': float(best_auc),
        'performance_level': performance_level,
        'total_datasets': len(all_results),
        'total_methods': len(avg_performance),
        'avg_performance': avg_performance.to_dict()
    }
    
    with open('/content/results_summary.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"\nResults saved to results_summary.pkl")
    print("Analysis complete.")

if __name__ == "__main__":
    main()
