# Can Unsupervised Learning Discover Stable Equity Themes?

A multi-horizon clustering study of U.S.-listed equities using daily return co-movement.

[Open the research notebook](ML_cluster.ipynb)

## Overview

This project investigates whether unsupervised machine learning can identify stable and economically meaningful equity groups using price behaviour alone.

Instead of assigning companies to predefined industries, the analysis groups stocks according to similarities in their standardised daily returns. Principal Component Analysis (PCA), K-Means, consensus clustering, hierarchical clustering and HDBSCAN are combined to examine both broad market groups and narrower empirical themes.

The central research question is:

> Can unsupervised learning identify equity groups that are economically interpretable and stable across different time horizons and modelling assumptions?

The results show that return co-movement contains recognisable economic structure, but the groups overlap and change as the estimation horizon expands. The resulting themes are therefore best viewed as an exploratory complement to conventional industry classifications.

## Research Pipeline

```text
Daily closing prices
        |
        v
Current price, market-cap and share-class filters
        |
        v
Missing-data treatment and daily returns
        |
        v
Per-stock standardisation and return clipping
        |
        v
Principal Component Analysis
        |
        v
K-Means and stability analysis
        |
        v
Consensus and hierarchical clustering
        |
        v
Subcluster refinement and final equity themes
```

## Data and Universe

The latest notebook run uses data from 13 October 2021 to 16 October 2026.

| Item | Value |
|---|---:|
| Raw observations | 3,537,911 |
| Initial symbols | 3,042 |
| Trading days available | 1,259 |
| One-year eligible symbols | 2,927 |
| Three-year eligible symbols | 2,754 |
| Five-year eligible symbols | 2,604 |
| Symbols common to all three windows | 2,602 |

The initial universe applies the following current filters:

- Primary share classes only
- Current market capitalisation of at least USD 300 million
- Current share price of at least USD 1
- Positive historical closing prices
- No more than 5% missing observations within the selected window

The universe is suitable for classifying currently investable securities, but it is not a fully point-in-time or survivorship-bias-free historical universe.

## Data Preparation

For each one-, three- and five-year window:

1. Stocks with more than 5% missing prices are removed.
2. Remaining gaps are forward-filled.
3. Closing prices are converted into daily percentage returns.
4. Each stock's returns are standardised using its own mean and standard deviation.
5. Standardised values are clipped at ±5 to reduce the influence of extreme observations.
6. The matrix is transposed so that each stock is one machine-learning observation.

Forward-filling can create artificial zero returns. The missing-data threshold reduces, but does not eliminate, this limitation.

## Dimensionality Reduction

PCA compresses hundreds of daily return features into a smaller set of common components.

| Window | Variance explained by PC1 | Approximate variance explained by first 20 PCs | Components above shuffled baseline |
|---|---:|---:|---:|
| 1 year | 8.6% | 40.8% | 31 |
| 3 years | 5.8% | 31.8% | 61 |
| 5 years | 5.4% | 30.2% | 66 |

The clustering models retain 20 components as a common, parsimonious representation. Sensitivity to alternative component counts is evaluated separately.

## Baseline K-Means Clustering

Candidate cluster counts from 4 to 30 are evaluated using the silhouette score.

| Window | Highest displayed silhouette score | Corresponding K |
|---|---:|---:|
| 1 year | Approximately 0.192 | 8 or 18 |
| 3 years | 0.218 | 20 |
| 5 years | 0.236 | 20 |

The preferred resolution differs across windows. A common value of `K=20` is used for comparison rather than being presented as a universally optimal number of clusters.

The modest silhouette scores indicate that equity relationships overlap rather than form perfectly separated categories.

## Robustness and Stability

### PCA and random-seed sensitivity

The number of retained PCA components is varied from 10 to 50. K-Means is also fitted from multiple random initialisations.

| Window | Average ARI across random seeds |
|---|---:|
| 1 year | 0.710 |
| 3 years | 0.801 |
| 5 years | 0.669 |

The broad structure is reasonably reproducible, although individual stocks can move between neighbouring groups.

### Stability across time horizons

| Comparison | Adjusted Rand Index |
|---|---:|
| 1 year vs 3 years | 0.555 |
| 1 year vs 5 years | 0.449 |
| 3 years vs 5 years | 0.515 |

Agreement is highest between the one- and three-year results and lowest between the one- and five-year results. The evidence does not clearly support the hypothesis that longer windows are inherently more stable.

## Economic Interpretation

Market capitalisation and SIC descriptions are added only after clustering and are not used as model features.

The return-based groups recover several recognisable economic categories, including:

- Regional banks
- REITs
- Energy producers and refiners
- Shipping companies
- Pharmaceuticals and biotechnology
- Metals and mining
- Semiconductors
- Insurance and utilities

The correspondence is imperfect because return similarity may also reflect market beta, interest-rate exposure, commodity sensitivity, geography, volatility or temporary market events.

## Consensus and Alternative Clustering

Consensus clustering repeats PCA and K-Means across different component counts, cluster counts and random seeds. Each pair of stocks receives a co-clustering probability based on how often the pair appears in the same group.

The consensus similarity is converted into a distance matrix:

```text
distance(i, j) = 1 - consensus_probability(i, j)
```

This distance is used for average-linkage hierarchical clustering.

| Comparison | Result |
|---|---:|
| Ward hierarchy vs one-year K-Means ARI | 0.525 |
| Consensus hierarchy vs one-year K-Means ARI | 0.700 |
| HDBSCAN dense groups | 11 |
| HDBSCAN observations classified as noise | 76.8% |

HDBSCAN identifies a small number of dense economic islands but classifies most stocks as noise. This suggests that some themes are highly coherent while the broader equity universe has less distinct boundaries.

## Core and Edge Membership

The consensus matrix is used to estimate membership strength. Stocks are classified as:

- **Core:** consensus score above 0.70
- **Normal:** consensus score from 0.40 to 0.70
- **Edge:** consensus score below 0.40

The one-year consensus scores range from approximately 0.054 to 1.000, with a mean of 0.601 and median of 0.609.

The current implementation averages each stock's co-clustering probability across all members of its selected cluster, including itself.

## Hierarchical Theme Construction

The final theme model expands the sample to all 2,927 eligible one-year stocks.

The consensus hierarchy is examined at 20, 40, 50, 60 and 80 groups:

- The 20-group level represents broad return sectors.
- The 80-group level represents narrower candidate themes.
- Large 80-level groups containing at least 40 stocks are refined with additional PCA and K-Means clustering.
- The subcluster count is selected using silhouette scores, with a preference for the smallest solution within 0.01 of the best score.

A candidate theme is retained when it:

- Contains at least three stocks
- Has average pairwise return correlation of at least 0.40

Weak subclusters are assigned to an `.other` category, while groups with fewer than three members are assigned to `orphan`.

The final procedure identifies **224 return-based themes across 2,927 stocks**.

## Main Findings

| Finding | Evidence | Interpretation |
|---|---|---|
| Return data contain economically meaningful structure | Clusters recover several familiar sectors and themes | Industry and factor exposures are reflected in daily co-movement |
| Groups are not sharply separated | Maximum silhouette scores range from approximately 0.192 to 0.236 | The equity universe cannot be represented by one definitive partition |
| Seed stability is reasonably strong | Average seed ARI ranges from 0.669 to 0.801 | Broad structure is reproducible, but memberships are not permanent |
| Cross-horizon agreement is moderate | ARI ranges from 0.449 to 0.555 | Cluster relationships change with the estimation horizon |
| Consensus improves cross-method agreement | Consensus hierarchy vs K-Means ARI is 0.700 | Repeated clustering captures a robust common structure |
| Membership confidence varies substantially | Core scores range from 0.054 to 1.000 | Some stocks are stable representatives while others lie near group boundaries |
| Hierarchical refinement improves interpretability | 224 themes are identified across 2,927 stocks | Broad groups can be divided into narrower empirical themes |

The results support the hypotheses that return data contain interpretable structure, that broad groups are reasonably robust to model settings, and that consensus clustering can distinguish core from ambiguous members. The hypothesis that longer windows are inherently more stable is not clearly supported.

## Output Files

The notebook produces two Excel workbooks:

### `clusters.xlsx`

Contains the 2,602-stock common universe, including:

- One-, three- and five-year K-Means labels
- One-year HDBSCAN island labels
- One-year consensus core scores
- Core, normal and edge roles
- One-to-five-year neighbour stability
- Current market capitalisation and SIC description

### `theme.xlsx`

Contains the 2,927-stock one-year universe, including:

- Broad 20-group sector
- Final theme identifier
- Provisional theme name based on the three largest members
- Theme size
- Average within-theme correlation
- Current market capitalisation and SIC description

## Limitations

1. **Survivorship bias:** The universe is constructed from currently available securities and may exclude companies that previously delisted.
2. **Look-ahead bias:** Current market capitalisation and share-class status are used to select securities for historical analysis.
3. **Current classifications:** SIC descriptions come from the latest available company records rather than point-in-time classifications.
4. **Missing-data treatment:** Forward-filled prices create artificial zero returns during missing observations.
5. **Time-period dependence:** Relationships observed from October 2021 to October 2026 may not persist in other market regimes.
6. **Parameter sensitivity:** PCA dimensions, cluster counts, minimum theme size and the correlation threshold influence the result.
7. **Return-only features:** Fundamentals, company descriptions, supply-chain links and news are excluded.
8. **Forced membership:** K-Means assigns every stock to a cluster even when membership is weak.
9. **Provisional naming:** Theme names based on the three largest members may not describe every constituent.
10. **No investment validation:** The study does not test whether the themes improve portfolio returns, risk forecasts or diversification.

## Repository Contents

```text
.
|-- README.md
|-- ML_cluster.ipynb
|-- clusters.xlsx       # generated output
`-- theme.xlsx          # generated output
```

## Requirements

The notebook uses:

- Python 3
- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn 1.3 or later
- openpyxl
- Jupyter Notebook
- A local `N_DB` database interface

Install the public Python dependencies with:

```bash
pip install numpy pandas matplotlib scipy "scikit-learn>=1.3" openpyxl jupyter
```

## Data Interface

The private `N_DB` module must provide a `query()` function and access to the following tables:

| Table | Required fields |
|---|---|
| `bars_daily` | `symbol`, `date`, `close` |
| `mv_metrics` | `symbol`, `name`, `market_cap`, `close`, `is_primary_class` |
| `universe_current` | `symbol`, `sic_desc` |

Raw market data, database credentials and the private data infrastructure are not included. Users must supply equivalent licensed data and adapt the database-loading cells to their environment.

## Running the Research

1. Install the required packages.
2. Configure `N_DB` or replace its queries with an equivalent data loader.
3. Open `ML_cluster.ipynb`.
4. Restart the kernel and run all cells in order.
5. Confirm that the generated summaries match the reported results before drawing conclusions.

Because the universe uses current market data and a rolling five-year window, results can change when the notebook is rerun on a later date.

## Future Work

Future work could:

- Use a fully point-in-time, survivorship-bias-free universe
- Remove broad market and sector factors before clustering
- Compare simple returns with log returns
- Validate themes using rolling and out-of-sample windows
- Evaluate cluster quality against external industry classifications
- Replace provisional ticker-based names with reviewed semantic labels
- Incorporate sector, fundamental, financial-report, supply-chain and news information
- Test whether the themes improve portfolio diversification or risk modelling

## Disclaimer

This project is provided for educational and research purposes only. It does not constitute financial advice, investment advice or a recommendation to trade any security.
