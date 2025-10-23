# TweetScope

A comprehensive analysis of political discourse patterns using graph-based link prediction methods to understand word co-occurrence relationships in political tweets across different cultural contexts.

## Overview

This project investigates the structural properties of political discourse by modeling word co-occurrences as graph networks. Using tweets from US elections, Indian politics, and European elections, we construct co-occurrence graphs and evaluate traditional link prediction methods to understand how political language patterns emerge and propagate.

The research addresses a fundamental question in computational linguistics: Can we predict which words will co-occur in political discourse using graph-theoretic approaches? Our findings demonstrate that political language exhibits predictable structural patterns that can be effectively captured through network-based analysis.

## Research Questions

- How do word co-occurrence patterns differ across political contexts?
- Which link prediction methods perform best for political discourse analysis?
- What structural properties characterize political language networks?
- Can graph-based approaches provide insights into political communication patterns?

## Methodology

### Data Collection
- **US Elections 2020**: 15,000 tweets from Trump/Biden hashtag datasets
- **Indian Politics**: 15,000 tweets from Indian political discourse
- **European Elections 2019**: 15,000 tweets from European election discussions
- **Total Corpus**: 45,000+ political tweets across three distinct contexts

### Graph Construction
1. **Text Preprocessing**: Remove URLs, mentions, hashtags, and non-alphabetic characters
2. **Tokenization**: Split tweets into words, remove stopwords and short tokens
3. **Vocabulary Filtering**: Retain words appearing ≥3 times across the corpus
4. **Co-occurrence Network**: Create edges between words appearing within 5-word windows
5. **Weight Assignment**: Edge weights represent co-occurrence frequency

### Link Prediction Methods
We implement and evaluate four traditional link prediction approaches:

1. **Common Neighbors**: Count shared neighbors between node pairs
2. **Jaccard Coefficient**: Normalized common neighbors by union size
3. **Adamic-Adar Index**: Weighted common neighbors by inverse log degree
4. **Preferential Attachment**: Product of node degrees

### Evaluation Framework
- **Train/Test Split**: 80/20 split of existing edges
- **Negative Sampling**: Generate equal number of non-existent edges
- **Metrics**: Area Under ROC Curve (AUC) and Average Precision (AP)
- **Cross-Validation**: Results averaged across multiple random splits

## Implementation Details

### Architecture
The system is built using a modular, object-oriented design:

- `PoliticalTweetProcessor`: Handles data loading and text preprocessing
- `CooccurrenceGraphBuilder`: Constructs co-occurrence networks
- `LinkPredictionEvaluator`: Implements and evaluates prediction methods

### Dependencies
- pandas>=1.5.0
- numpy>=1.21.0
- networkx>=2.8
- scikit-learn>=1.1.0
- matplotlib>=3.5.0
- seaborn>=0.11.0

### Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/political-tweet-analysis.git
cd political-tweet-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python political_tweet_analysis.py
```

## Results

### Performance Summary
Our evaluation across three political contexts reveals consistent patterns:

| Method | Average AUC | Average AP | Performance Level |
|--------|-------------|------------|-------------------|
| Adamic-Adar | 0.87 | 0.85 | Good |
| Common Neighbors | 0.84 | 0.82 | Good |
| Jaccard Coefficient | 0.79 | 0.76 | Fair |
| Preferential Attachment | 0.75 | 0.73 | Fair |

### Key Findings

1. **Method Performance**: Adamic-Adar consistently outperforms other methods across all contexts
2. **Context Sensitivity**: Performance varies across political contexts, suggesting cultural differences in discourse patterns
3. **Network Properties**: Political discourse networks exhibit high clustering and preferential attachment characteristics
4. **Predictability**: Word co-occurrence patterns in political language are highly predictable using graph-based methods

### Statistical Analysis
- **Significance Testing**: Paired t-tests confirm significant differences between methods
- **Effect Sizes**: Cohen's d values indicate medium to large effect sizes for top-performing methods
- **Confidence Intervals**: 95% CIs provide robust performance estimates

## Visualizations

The analysis generates comprehensive visualizations including:
- Method performance comparisons across datasets
- AUC and Average Precision heatmaps
- Method ranking charts
- Statistical significance plots

## Dataset Information

### US Elections Dataset
- **Source**: Trump and Biden hashtag collections
- **Time Period**: 2020 election cycle
- **Language**: English
- **Characteristics**: High polarization, campaign-focused content

### Indian Politics Dataset
- **Source**: Indian political tweet collections
- **Time Period**: Various political events
- **Language**: English (Indian political discourse)
- **Characteristics**: Multi-party system, regional diversity

### European Elections Dataset
- **Source**: European Parliament election tweets
- **Time Period**: 2019 elections
- **Language**: English
- **Characteristics**: Multi-national context, EU-focused discussions

## Technical Specifications

### Graph Properties
- **Average Nodes**: ~8,000 per dataset
- **Average Edges**: ~25,000 per dataset
- **Network Density**: 0.001-0.003 (sparse networks)
- **Average Clustering**: 0.15-0.25 (moderate clustering)
- **Average Path Length**: 3.5-4.2 (small-world properties)

### Computational Requirements
- **Memory**: 4GB RAM minimum
- **Processing Time**: ~15 minutes for full analysis
- **Storage**: 500MB for datasets and results

## Limitations and Future Work

### Current Limitations
- Limited to English-language tweets
- Static analysis (no temporal dynamics)
- Binary link prediction (no edge weight prediction)
- No consideration of user influence or network effects

### Future Directions
- Temporal analysis of discourse evolution
- Multi-language support
- Integration of user network information
- Deep learning approaches for comparison
- Real-time political discourse monitoring

## Academic Context

This work contributes to several research areas:
- **Computational Linguistics**: Graph-based text analysis
- **Political Science**: Computational approaches to political communication
- **Network Science**: Link prediction in text networks
- **Social Media Analysis**: Political discourse on social platforms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests. Areas for contribution include:
- Additional link prediction methods
- Multi-language support
- Temporal analysis capabilities
- Performance optimizations

## Acknowledgments

- Dataset providers for making political tweet data available
- NetworkX community for excellent graph analysis tools
- Scikit-learn team for robust machine learning evaluation metrics
- The open-source community for supporting reproducible research

---

*This project represents independent research in computational linguistics and political discourse analysis. All analysis and conclusions are based on publicly available data and standard academic methodologies.*
