

This project leverages a series of experiments to extract, process, and model financial sentiment data from various textual and market data sources. The goal is to derive insights from both news articles and Reddit WallStreetBets (WSB) comments, and use these signals to better understand the impact financial news has on retail investor behaviour. Please note that some code files have not been included as they involve mentions and processing of DWS internal data. Moreover, certain code files have also not been included as I believe they possess commercialisation potential subject to certain enhancements; however, the code files present should provide a good overview of the general analysis workflow. 

---

## Experiment 1: Intelligent Data Curation for Financial Text

### Files

- **article_preprocessing.py**  
  This script automates the end-to-end curation of financial articles by:
  - Parsing ProQuest-exported text files.
  - Cleaning and normalizing text (removing HTML and extraneous elements while preserving key financial terms).
  - Filtering articles for relevance using spaCy’s Named Entity Recognition and keyword matching.
  - Optionally reducing noise via TF-IDF vectorization and LDA topic modeling.
  - Integrating financial market data by fetching trading volumes from Yahoo Finance.  
  Together, these processes provide a comprehensive framework for consolidating and analyzing text-based news alongside market metrics.

- **finbert_finetuning.ipynb**  
  This Jupyter Notebook leverages a preprocessed financial sentiment dataset to fine-tune a FinBERT model for sequence classification into three labels. The workflow includes:
  - Loading and cleaning CSV data.
  - Encoding labels and splitting the dataset into training, validation, and test sets.
  - Tokenizing text using the FinBERT tokenizer.
  - Preparing the data in a PyTorch-compatible format.
  - Defining training parameters (learning rate, batch size, epochs) and using an accuracy-based evaluation metric.
  - Utilizing Hugging Face's Trainer API to train, evaluate, and save the fine-tuned model for future applications.

- **finbert_sentiment_analysis.ipynb**  
  This notebook calculates sentiments for financial article titles by:
  - Loading a preprocessed, zipped CSV dataset from Google Drive.
  - Combining article titles and content into a single text field.
  - Using the fine-tuned FinBERT model to predict sentiment labels and scores.
  - Tokenizing the combined text, processing it on a GPU or CPU, and applying softmax to assign classifications ("negative," "neutral," or "positive").
  - Compiling the results with corresponding company names and dates into a DataFrame and saving to CSV for further analysis.

---

## Experiment 2: Quantifying Market Reactions: Reddit Sentiment Dynamics and Options data

### Files

- **reddit_preprocessing.sh**  
  A Bash script that automates the extraction of ticker-related comments from a WallStreetBets input file by:
  - Iterating over a predefined list of ticker symbols.
  - Using GNU grep to filter lines containing each ticker.
  - Dynamically determining the number of CPU cores (on macOS) to parallelize the search.
  - Directing the output for each ticker into its respective text file.  
  This streamlined process facilitates subsequent data segregation and analysis.

- **reddit_json_analysis.ipynb**  
  This comprehensive notebook orchestrates an extensive analysis of financial sentiment by:
  - Installing necessary libraries and downloading data.
  - Applying advanced text cleaning techniques using NLTK, spaCy, and regex-based methods.
  - Extracting sentiment scores from both news titles and Reddit comments using transformer-based models and VADER (enhanced with a custom lexicon).
  - Merging these sentiment scores with market metrics (e.g., trading volume, options data) per ticker.
  - Performing correlation analyses (Pearson, Spearman), lagged assessments, Granger causality tests, and topic modeling (LDA) with visualizations to explore relationships between sentiment and market activity.

- **reddit_sentiment_summary_analysis.ipynb**  
  This notebook extracts sentiment from preprocessed Reddit WSB comments that have been mapped to respective stock tickers using VADER (enhanced with WSB-specific terms). The Notebook:
  - Installs the required libraries and configures the VADER analyzer.
  - Processes CSV files from a designated input directory to calculate compound sentiment scores for each comment.
  - Integrates these scores with financial articles by mapping them to common dates.
  - Performs further analyses such as correlation calculations, scatter and density plots, and case counts comparing positive versus negative sentiment instances to assess the alignment between Reddit investor sentiment and traditional market sentiment.

- **options_analysis.ipynb**  
  This notebook implements a multi-step pipeline for merging, cleaning, and analyzing financial options data alongside sentiment metrics by:
  - Merging daily and monthly options volume data (from CBOE) with advanced volume metrics.
  - Renaming and cleaning data columns and calculating percentage spike values.
  - Joining the options data with financial articles enriched by Reddit sentiment scores.
  - Conducting exploratory data analysis (visualizations, summary statistics) to compare data distributions across publication dates.
  - Applying advanced predictive modeling techniques (linear regression, random forest, gradient boosting, neural networks, and LSTM) to forecast options volume spikes based on sentiment scores.
  - Aggregating correlations by company and sector to generate comprehensive insights into the relationship between market sentiment and options trading activity.

---

## Experiment 3: Preiciting retail Investor Ssentiment - News Driven Classification and Momentum Analysis 

### File

- **exp3.ipynb**  
  This notebook implements an extensive machine learning pipeline on a financial dataset that combines as input:
  - Sentiment measures from both news and Reddit WallStreetBets comments.
  - Derived momentum indicators.
  
  It employs a variety of modeling techniques including:
  - **Multi-Output Regression with PyTorch,** optimized by Optuna for hyperparameter tuning.
  - **Traditional Regressors:** Linear regression, Random Forest, Gradient Boosting, and LSTM networks.
  - **Classification Models:** Logistic Regression, Random Forest, XGBoost, and automated ML via TPOT to predict a binary RSI label computed from weighted sentiment metrics.
  
  Additionally, the notebook:
  - Incorporates feature importance and explainability analyses using SHAP.
  - Fetches and merges trading data from Yahoo Finance.
  - Trains sector-specific models based on a ticker-to-sector mapping.
  
  Together, these methods provide a comprehensive framework for assessing and predicting retail investor sentiment based on financial-news-sentiment-derived insights.

---


