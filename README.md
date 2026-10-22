# Twitter US Airline Sentiment Analysis

Classifying tweets about major U.S. airlines as **negative**, **neutral**, or **positive** using text mining and machine learning (TF-IDF + Linear SVC).

Data Mining course project, Tunis Business School, academic year 2025–2026.

## Overview

The goal is twofold: build a model that predicts tweet sentiment, and understand what drives customer dissatisfaction (which words, airlines, and complaint categories are most associated with negative sentiment).

## Dataset

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (`Tweets.csv`), about 14,600 tweets labelled by sentiment, with the airline, label confidence, negative reason, creation time, and retweet count.

The classes are imbalanced: **62.7% negative**, 21.2% neutral, 16.1% positive.

> The dataset is not included in this repository. Download `Tweets.csv` from Kaggle and place it in the same folder as `NLP.py`.

## Approach

**Preprocessing**
- Removed low-confidence labels (`airline_sentiment_confidence` < 0.60)
- Cleaned text: lowercasing, removal of URLs, mentions, punctuation, HTML entities, and numbers
- Expanded contractions, removed stopwords (keeping negations such as *not* and *never*), and lemmatized
- Dropped tweets left empty after cleaning, then made a stratified train / validation / test split

**Features**
- TF-IDF with unigrams and bigrams, with the vocabulary limited to informative terms
- Hand-crafted features: character count, word count, punctuation counts, emoji presence, hashtag count, mention count, and a negation indicator (standardized and combined with TF-IDF)

**Models compared**
- Logistic Regression
- Complement Naive Bayes
- Linear SVC

`class_weight="balanced"` was used for Logistic Regression and Linear SVC to handle class imbalance. Models were compared with 5-fold cross-validation on the training data and then on the validation set. **Linear SVC** gave the best balance across the three classes and was selected as the final model.

## Results

Final Linear SVC performance on the held-out test set:

| Class    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| negative | 0.86      | 0.88   | 0.87     |
| neutral  | 0.66      | 0.64   | 0.65     |
| positive | 0.78      | 0.74   | 0.76     |

![Test set confusion matrix](test_confusion_matrix.png)

## Key findings

- **Negative tweets are the easiest to classify; neutral tweets are the hardest.** 30.3% of neutral tweets are predicted as negative, because neutral airline tweets often use complaint-like vocabulary.
- **Customer service issues** are the most frequent reason for negative sentiment (2,910 tweets), followed by late flights (1,665) and cancelled flights (847).
- **US Airways** has the highest share of negative tweets, followed by American and United. **Virgin America** has the most balanced sentiment and the largest positive share.
- Negative tweets tend to be longer than neutral and positive ones. Positive tweets center on gratitude (*thanks*, *great*, *love*, *awesome*).

## Getting started

```bash
git clone https://github.com/Dhia0184/twitter-airline-sentiment-analysis.git
cd twitter-airline-sentiment-analysis

python -m venv .venv
.venv\Scripts\activate        # Windows (on macOS/Linux: source .venv/bin/activate)

pip install -r requirements.txt
python NLP.py
```

The script downloads the required NLTK data (stopwords and WordNet) on first run, so an internet connection is needed. It saves the figures (class distribution, sentiment by airline, confusion matrices, classification reports, and others) as PNG files in the working folder.

## Repository structure

```
├── NLP.py              # full pipeline: EDA, preprocessing, features, models, evaluation
├── requirements.txt    # Python dependencies
├── README.md
└── Tweets.csv          # not included, download from Kaggle
```

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, NLTK, Matplotlib, Seaborn

## Team

Group project by Mahdi Haddad, Mohamed Dhia Ben Kilani, Nessrine Nebli, and Wassef Bellila.

AI tools were used to help generate the code, as stated in the project report.
