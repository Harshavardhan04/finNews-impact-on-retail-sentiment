#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
Intelligent Data Curation for Financial Text
------------------------------------------------------------------------------

This script implements an end-to-end workflow for:
    1) Gathering and parsing ProQuest-exported financial articles 
       (including subfolders for each company).
    2) Cleaning and normalizing text (HTML removal, preserving domain terms).
    3) Named Entity Recognition (NER)-based and keyword-based relevance filtering 
       to isolate articles pertinent to target companies or finance.
    4) Noise reduction using LDA topic modeling (optional final filter).
    5) A helper function to fetch trading volume data from Yahoo Finance.

==============================================================================
"""

import os
import re
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# (Optional) For TF-IDF and LDA-based topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# (Optional) For NER-based filtering
import spacy
from fuzzywuzzy import fuzz
from tqdm import tqdm

# For Yahoo Finance trading volume
import yfinance as yf
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. UTILITY FUNCTION: FETCH TRADING VOLUME FROM YAHOO FINANCE
# ---------------------------------------------------------------------------
def fetch_trading_volume_data(ticker, date_str):
    """
    Fetch trading volume for a specified ticker on a specified date 
    from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        The stock symbol (e.g., 'AAPL' for Apple).
    date_str : str
        The date (YYYY-MM-DD) to fetch the trading volume for.

    Returns
    -------
    trading_volume : float or int
        The trading volume on the specified date, or None if not found.

    Usage Example:
    --------------
    volume = fetch_trading_volume_data("TSLA", "2023-01-15")
    """
    try:
        data = yf.download(ticker, start=date_str, end=date_str)
        if not data.empty:
            trading_volume = data['Volume'].values[0]
            return trading_volume
        else:
            print(f"No data found for ticker={ticker} on date={date_str}.")
            return None
    except Exception as e:
        print(f"Error fetching trading volume for {ticker} on {date_str}: {e}")
        return None

# ---------------------------------------------------------------------------
# 2. PARSING PROQUEST TEXT FILES TO PRODUCE A PROCESSED CSV
#    (If you already have your articles in CSV, skip this section)
# ---------------------------------------------------------------------------
def parse_proquest_folder(proquest_folder, output_csv="processed_proquest_articles_with_dates.csv"):
    """
    Recursively walks through the ProQuest folder and subfolders,
    parsing *.txt files, extracting fields such as Title, Content, and Publication Date,
    and organizes them into a DataFrame saved as output_csv.

    Parameters
    ----------
    proquest_folder : str
        Path to the ProQuest directory containing subfolders with .txt files.
    output_csv : str
        Name of the output CSV file to store the parsed data.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns [Company, Document Number, Title, Content, Date].
    """

    data = []
    doc_number_tracker = {}  # per-company numbering

    def extract_field(text, start_marker, end_marker):
        try:
            start_index = text.index(start_marker) + len(start_marker)
            end_index = text.index(end_marker, start_index)
            return text[start_index:end_index]
        except ValueError:
            return None

    def extract_field_until(text, start_marker, stop_markers):
        try:
            start_index = text.index(start_marker) + len(start_marker)
            end_index = len(text)
            # find earliest occurrence among stop markers
            for marker in stop_markers:
                marker_index = text.find(marker, start_index)
                if marker_index != -1:
                    end_index = min(end_index, marker_index)
            return text[start_index:end_index]
        except ValueError:
            return None

    def extract_date(text):
        try:
            # e.g. "Publication date: Oct 30, 2024"
            date_marker = "Publication date: "
            start_index = text.index(date_marker) + len(date_marker)
            end_index = text.index("\n", start_index)
            raw_date = text[start_index:end_index].strip()
            # parse date
            parsed_date = datetime.strptime(raw_date, "%b %d, %Y")
            return parsed_date.strftime("%Y-%m-%d 00:00:00")
        except (ValueError, IndexError):
            return None

    def process_txt_file(file_path, company_name):
        nonlocal data
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            articles = content.split("____________________________________________________________")
            for article in articles:
                if "Document" in article:
                    title = extract_field(article, "Title: ", "\n")
                    full_text = extract_field_until(article, "Full text: ", 
                                                    ["Subject:", "Business indexing term:"])
                    date = extract_date(article)
                    if title and full_text and date:
                        if company_name not in doc_number_tracker:
                            doc_number_tracker[company_name] = 1
                        data.append({
                            'Company': company_name,
                            'Document Number': doc_number_tracker[company_name],
                            'Title': title.strip(),
                            'Content': full_text.strip(),
                            'Date': date
                        })
                        doc_number_tracker[company_name] += 1

    # walk through subfolders
    for root, _, files in os.walk(proquest_folder):
        for file in sorted(files):
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                # folder name as company name, or file name if top-level
                company_name = os.path.basename(root) if root != proquest_folder else os.path.splitext(file)[0]
                process_txt_file(file_path, company_name)

    # DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Processed {len(df)} articles from '{proquest_folder}'. Saved to '{output_csv}'.")
    return df

# ---------------------------------------------------------------------------
# 3. CLEANING AND NORMALIZING TEXT: Removing HTML, preserving domain terms
# ---------------------------------------------------------------------------
def clean_proquest_articles(input_file, output_file="cleaned_proquest_articles.csv"):
    """
    Cleans and normalizes text content from the input_file CSV.

    Parameters
    ----------
    input_file : str
        Path to the CSV with columns ['Content', 'Company', ...].
    output_file : str
        Name of the output CSV file after cleaning.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with new column 'Cleaned_Content'.
    """

    df = pd.read_csv(input_file)
    if 'Content' not in df.columns:
        print("Error: 'Content' column not found in the dataset.")
        return None

    # domain-specific financial terms
    financial_terms = [
        "revenue", "profit", "loss", "shares", "stock", "market",
        "billion", "million", "quarter", "earnings", "growth",
        "company", "forecast", "dividend", "equity", "capital"
    ]
    financial_terms_pattern = re.compile(r'\b(?:' + '|'.join(financial_terms) + r')\b')

    def clean_text_financial(text):
        if not isinstance(text, str):
            return ""
        preserved_terms = financial_terms_pattern.findall(text.lower())
        text = re.sub(r'<.*?>', '', text)  # remove HTML
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
        text = re.sub(r'\S+\.com\b|\S+\.net\b|\S+\.org\b|\S+\.edu\b|\S+\.gov\b', '', text)
        text = re.sub(r'[^\w\s.%]', '', text)  # keep periods in decimal numbers
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        # re-append preserved terms to ensure they remain
        preserved_text = " ".join(set(preserved_terms))
        return text + " " + preserved_text

    df['Cleaned_Content'] = df['Content'].apply(clean_text_financial)
    df.to_csv(output_file, index=False)
    print(f"Text cleaning complete. {len(df)} rows saved to '{output_file}'.")
    return df

# ---------------------------------------------------------------------------
# 4. NER-BASED + KEYWORD-BASED RELEVANCE FILTERING
# ---------------------------------------------------------------------------
def filter_relevant_articles(
    input_file, 
    relevant_output="relevant_proquest_articles.csv",
    non_relevant_output="non_relevant_proquest_articles.csv"
):
    """
    Filters articles for relevance using spaCy-based NER matching against 
    target company names plus fallback industry keywords.

    Parameters
    ----------
    input_file : str
        CSV file containing columns ['Cleaned_Content','Company',...].
    relevant_output : str
        CSV file to save relevant articles.
    non_relevant_output : str
        CSV file to save non-relevant articles.

    Returns
    -------
    relevant_df : pandas.DataFrame
        DataFrame containing rows deemed relevant.
    non_relevant_df : pandas.DataFrame
        DataFrame containing rows deemed irrelevant.
    """

    df = pd.read_csv(input_file)
    if 'Cleaned_Content' not in df.columns:
        print("Error: 'Cleaned_Content' column not found.")
        return None, None

    # load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Predefined list of suffixes to remove from company names
    suffixes_to_remove = [" co", " corp", " inc", " plc", " incorporated", " corporation"]
    def clean_company_name(name):
        name = name.lower()
        for sfx in suffixes_to_remove:
            name = name.replace(sfx, "")
        return name.strip()

    if 'Company' in df.columns:
        target_companies = [clean_company_name(c) for c in df['Company'].dropna().unique()]
    else:
        print("Warning: 'Company' column not found. Using empty target list.")
        target_companies = []

    # threshold for partial matches
    SIMILARITY_THRESHOLD = 70
    # fallback industry keywords
    industry_keywords = ["revenue", "merger", "acquisition", "profit", "loss", "growth"]

    def is_relevant(text):
        if not isinstance(text, str):
            return False
        doc = nlp(text)
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
        # fuzzy matching for each recognized entity
        for company in target_companies:
            for entity in entities:
                if fuzz.partial_ratio(company, entity) >= SIMILARITY_THRESHOLD:
                    return True
        # fallback check for domain keywords
        for keyword in industry_keywords:
            if keyword in text.lower():
                return True
        return False

    tqdm.pandas(desc="Filtering for Relevance")
    df['Is_Relevant'] = df['Cleaned_Content'].progress_apply(is_relevant)

    relevant_df = df[df['Is_Relevant']]
    non_relevant_df = df[~df['Is_Relevant']]

    relevant_df.to_csv(relevant_output, index=False)
    print(f"Relevance filtering: {len(relevant_df)} articles --> '{relevant_output}'")

    non_relevant_df.to_csv(non_relevant_output, index=False)
    print(f"Non-relevant: {len(non_relevant_df)} articles --> '{non_relevant_output}'")

    return relevant_df, non_relevant_df

# ---------------------------------------------------------------------------
# 5. NOISE REDUCTION USING LDA TOPIC MODELING
# ---------------------------------------------------------------------------
def apply_lda_noise_reduction(
    input_file,
    output_file="noise_reduced_proquest_articles.csv",
    keywords_output_file="topic_keywords.csv",
    num_topics=5,
    max_features=5000
):
    """
    Applies TF-IDF vectorization and LDA for topic modeling on cleaned text,
    then filters out articles that do not belong to relevant topics (if desired).

    Parameters
    ----------
    input_file : str
        CSV file with a 'Cleaned_Content' column.
    output_file : str
        CSV file to save final noise-reduced dataset.
    keywords_output_file : str
        CSV file to save the top topic keywords.
    num_topics : int
        Number of topics for LDA.
    max_features : int
        Maximum vocabulary size for TF-IDF.

    Returns
    -------
    filtered_df : pandas.DataFrame
        DataFrame of articles assigned to the relevant LDA topics.
    """

    df = pd.read_csv(input_file)
    if 'Cleaned_Content' not in df.columns:
        print("Error: 'Cleaned_Content' column not found.")
        return None

    custom_stopwords = [
        "said", "mr", "new", "company", "companies", "shares", 
        "tesla", "amazon", "walmart", "apple", "bank", "banks", 
        "netflix", "morgan", "jp", "citigroup", "wells", "billion"
    ]

    # TF-IDF Vectorization
    print("TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=custom_stopwords,
        ngram_range=(1,2)  # bigrams for additional context
    )
    dtm = vectorizer.fit_transform(tqdm(df['Cleaned_Content'], desc="TF-IDF Progress"))

    # LDA Topic Modeling
    print("Applying LDA (num_topics={})...".format(num_topics))
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_topics = lda.fit_transform(dtm)

    # Assign dominant topic
    df['Dominant_Topic'] = np.argmax(lda_topics, axis=1)

    # Extract top keywords for each topic
    words = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-11:-1]  # top 10
        top_words = [words[i] for i in top_indices]
        topic_keywords[f"Topic_{topic_idx}"] = top_words

    # Save topic keywords
    kw_df = pd.DataFrame.from_dict(topic_keywords, orient='index')
    kw_df.to_csv(keywords_output_file, index=True)
    print(f"Topic keywords saved to '{keywords_output_file}'")

    # In principle, we might choose relevant topics after manual inspection.
    # For demonstration, let's keep *all* topics or define a subset if needed:
    relevant_topics = list(range(num_topics))  # or choose subset
    filtered_df = df[df['Dominant_Topic'].isin(relevant_topics)]
    filtered_df.to_csv(output_file, index=False)
    print(f"LDA noise reduction complete: {len(filtered_df)}/{len(df)} articles remain. Saved to '{output_file}'.")
    return filtered_df


