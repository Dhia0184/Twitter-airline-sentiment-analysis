import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the data
df = pd.read_csv("Tweets.csv")

#  Missing values
missing = df.isnull().sum()
print(f"\nMissing values:\n{missing[missing > 0]}")

# Class distribution 
print(f"\nSentiment distribution:\n{df['airline_sentiment'].value_counts()}")
print(f"\nAs percentages:\n{df['airline_sentiment'].value_counts(normalize=True).mul(100).round(1)}")

# Columns to keep
cols_of_interest = [
    'airline_sentiment',      
    'airline_sentiment_confidence',
    'text',                   
    'airline',                
    'negativereason',         
    'tweet_created',
    'retweet_count'
]
df = df[cols_of_interest].copy()

# Sample tweets per class 
for label in ['positive', 'neutral', 'negative']:
    print(f"\n── {label.upper()} ──")
    samples = df[df['airline_sentiment'] == label]['text'].sample(3, random_state=42)
    for tweet in samples:
        print(f"  • {tweet}")



fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Counts
sentiment_counts = df['airline_sentiment'].value_counts()
colors = {'negative': '#E24B4A', 'neutral': '#EF9F27', 'positive': '#1D9E75'}
color_list = [colors[s] for s in sentiment_counts.index]

axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=color_list, edgecolor='white', linewidth=0.5)
axes[0].set_title('Sentiment counts', fontsize=13)
axes[0].set_ylabel('Number of tweets')
for i, (label, count) in enumerate(sentiment_counts.items()):
    axes[0].text(i, count + 100, str(count), ha='center', fontsize=11)

# Percentages as pie
pct = sentiment_counts / sentiment_counts.sum() * 100
axes[1].pie(pct, labels=[f"{s}\n{p:.1f}%" for s, p in zip(pct.index, pct)],
            colors=color_list, startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Sentiment split', fontsize=13)

# Overall title and layout
plt.suptitle('Class distribution — Twitter US Airline Sentiment', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(sentiment_counts)
print(f"\nImbalance ratio (negative:positive): {sentiment_counts['negative']/sentiment_counts['positive']:.1f}x")

airline_sentiment = df.groupby(['airline', 'airline_sentiment']).size().unstack(fill_value=0)
airline_sentiment_pct = airline_sentiment.div(airline_sentiment.sum(axis=1), axis=0) * 100

ax = airline_sentiment_pct[['negative', 'neutral', 'positive']].plot(
    kind='bar', figsize=(12, 5), color=['#E24B4A', '#EF9F27', '#1D9E75'],
    edgecolor='white', linewidth=0.5, width=0.7
)
ax.set_title('Sentiment distribution by airline (%)', fontsize=13)
ax.set_ylabel('Percentage of tweets')
ax.set_xlabel('')
ax.legend(title='Sentiment', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('sentiment_by_airline.png', dpi=150, bbox_inches='tight')
plt.show()
print(airline_sentiment_pct.round(1))

# Volume of tweets per airline
tweet_counts = df['airline'].value_counts()

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(tweet_counts.index, tweet_counts.values, color='#378ADD', edgecolor='white', linewidth=0.5)
ax.set_title('Tweet volume per airline', fontsize=13)
ax.set_xlabel('Number of tweets')
for bar, count in zip(bars, tweet_counts.values):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=10)
plt.tight_layout()
plt.savefig('tweet_volume.png', dpi=150, bbox_inches='tight')
plt.show()

# Confidence distribution by sentiment class
fig, ax = plt.subplots(figsize=(9, 4))
for sentiment, color in colors.items():
    subset = df[df['airline_sentiment'] == sentiment]['airline_sentiment_confidence']
    ax.hist(subset, bins=30, alpha=0.6, color=color, label=sentiment, edgecolor='white')

ax.axvline(0.6, color='gray', linestyle='--', linewidth=1, label='threshold (0.6)')
ax.set_title('Labeling confidence by sentiment class', fontsize=13)
ax.set_xlabel('Confidence score')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Check how many low-confidence labels we have
low_conf = (df['airline_sentiment_confidence'] < 0.6).sum()
print(f"Low-confidence labels (<0.6): {low_conf} ({low_conf/len(df)*100:.1f}% of dataset)")

df['tweet_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for sentiment, color in colors.items():
    subset = df[df['airline_sentiment'] == sentiment]
    axes[0].hist(subset['tweet_length'], bins=40, alpha=0.5, color=color, label=sentiment)
    axes[1].hist(subset['word_count'], bins=30, alpha=0.5, color=color, label=sentiment)

axes[0].set_title('Character length by sentiment', fontsize=13)
axes[0].set_xlabel('Characters')
axes[1].set_title('Word count by sentiment', fontsize=13)
axes[1].set_xlabel('Words')
for ax in axes:
    ax.legend()
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('tweet_lengths.png', dpi=150, bbox_inches='tight')
plt.show()

print(df.groupby('airline_sentiment')[['tweet_length', 'word_count']].mean().round(1))
# Reasons for negative label
neg_reasons = df['negativereason'].value_counts().dropna()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(neg_reasons.index, neg_reasons.values, color='#E24B4A', edgecolor='white', linewidth=0.5)
ax.set_title('Reasons for negative sentiment', fontsize=13)
ax.set_xlabel('Count')
ax.invert_yaxis()
for bar, count in zip(bars, neg_reasons.values):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('negative_reasons.png', dpi=150, bbox_inches='tight')
plt.show()
print(neg_reasons)

# Most frequent words per sentiment class


def get_top_words(texts, n=20):
    words = []
    for text in texts:
        # basic clean for visualization
        text = re.sub(r'http\S+|@\w+|[^a-zA-Z\s]', '', text.lower())
        words.extend(text.split())
    stopwords = {'the','a','to','i','and','is','in','of','for','on',
                 'my','me','you','your','this','it','was','are','have',
                 'be','with','at','so','we','but','not','had','has','an',
                 'they','that','our','can','got','get','will','just','us'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(words).most_common(n)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (sentiment, color) in zip(axes, colors.items()):
    subset = df[df['airline_sentiment'] == sentiment]['text']
    top_words = get_top_words(subset)
    words, counts = zip(*top_words)
    ax.barh(list(words)[::-1], list(counts)[::-1], color=color, edgecolor='white', linewidth=0.5)
    ax.set_title(f'Top words — {sentiment}', fontsize=12)
    ax.set_xlabel('Frequency')

plt.suptitle('Most frequent words by sentiment class', fontsize=14)
plt.tight_layout()
plt.savefig('top_words.png', dpi=150, bbox_inches='tight')
plt.show()

import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')



# Drop low-confidence labels
df = df[df['airline_sentiment_confidence'] >= 0.6].copy()
print(f"Rows after confidence filter: {len(df)}")

# Drop columns we won't use for training
df = df.drop(columns=['airline_sentiment_confidence', 'negativereason', 'tweet_created'])

# Reset index
df = df.reset_index(drop=True)
print(df.shape)

# Text cleaning function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

negations = {'no', 'not', 'nor', 'neither', 'never', 'nobody',
             'nothing', 'nowhere', 'hardly', 'barely', 'scarcely'}
stop_words = stop_words - negations

def clean_tweet(text):
    # 1. Lowercase
    text = text.lower()
    # 1b. Normalize unicode punctuation
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # strip non-ASCII

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remove @mentions, HTML entities, and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'#\w+', '', text)

    # 4. Expand common contractions
    contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am"
    }
    for pattern, replacement in contractions.items():
        text = text.replace(pattern, replacement)

    # 5. Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
     # 6. Remove numbers
    text = re.sub(r'\d+', '', text)# 7. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()# 8. Tokenize
    tokens = text.split()# 9. Remove stopwords (keeping negations)
    tokens = [t for t in tokens if t not in stop_words]# 10. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # 11. Remove very short tokens (single letters)
    tokens = [t for t in tokens if len(t) > 1]

    return ' '.join(tokens)

#comparison of original vs cleaned text
df['cleaned_text'] = df['text'].apply(clean_tweet)

# Side-by-side comparison
print("── Before vs After ──────────────────────────────────────")
for i in [0, 100, 500, 1000, 3000]:
    print(f"\nOriginal : {df['text'].iloc[i]}")
    print(f"Cleaned  : {df['cleaned_text'].iloc[i]}")

# Sanity check
# Check for empty strings after cleaning
empty = (df['cleaned_text'].str.strip() == '').sum()
print(f"Empty tweets after cleaning: {empty}")

# Check cleaned length distribution
df['cleaned_length'] = df['cleaned_text'].str.split().str.len()
print(f"\nCleaned word count stats:")
print(df.groupby('airline_sentiment')['cleaned_length'].describe().round(1))

# Flag very short tweets
very_short = df[df['cleaned_length'] <= 2]
print(f"\nTweets with ≤2 words after cleaning: {len(very_short)}")
print(very_short[['text', 'cleaned_text', 'airline_sentiment']].head(10))
# Label encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['airline_sentiment'])

# Check the mapping
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping:", mapping)
# {'negative': 0, 'neutral': 1, 'positive': 2}

# Stratified train/val/test split (To preserve class distribution)
# Redo the split, carrying raw text along
X_train, X_temp, y_train, y_temp, raw_train, raw_temp = train_test_split(
    df['cleaned_text'], df['label'], df['text'],
    test_size=0.2, random_state=42, stratify=df['label']
)

X_val, X_test, y_val, y_test, raw_val, raw_test = train_test_split(
    X_temp, y_temp, raw_temp,
    test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Verify stratification held
for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    counts = pd.Series(y_split).value_counts(normalize=True).sort_index()
    print(f"\n{split_name} class balance:")
    for label_idx, pct in counts.items():
        label_name = label_encoder.classes_[label_idx]
        print(f"  {label_name}: {pct*100:.1f}%")

df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
print(f"Rows after dropping empties: {len(df)}")

#TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=10000,     
    ngram_range=(1, 2),     
    min_df=3,               
    max_df=0.95,            
    sublinear_tf=True       
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}")
print(f"Sparsity: {100 * (1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.1f}%")

# Peek at top terms
feature_names = tfidf.get_feature_names_out()
print(f"\nSample features: {feature_names[:20].tolist()}")
print(f"Sample bigrams: {[f for f in feature_names if ' ' in f][:15]}")

# Handcrafted features
def extract_handcrafted_features(texts_raw):
    features = []
    for text in texts_raw:
        f = {}

        # Length signals
        f['char_count']      = len(text)
        f['word_count']      = len(text.split())
        f['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0

        # Punctuation signals
        f['exclamation_count'] = text.count('!')
        f['question_count']    = text.count('?')
        f['ellipsis_count']    = text.count('...')
        f['caps_ratio']        = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Sentiment-adjacent patterns
        f['has_emoji']         = int(bool(re.search(r'[\U0001F300-\U0001FFFF]', text)))
        f['has_url']           = int(bool(re.search(r'http\S+', text)))
        f['mention_count']     = len(re.findall(r'@\w+', text))
        f['hashtag_count']     = len(re.findall(r'#\w+', text))

        # Negation signal 
        negation_words = ['not', 'never', 'no', 'cannot', 'cant', "can't",
                          'nothing', 'nobody', 'nowhere', 'hardly', 'barely']
        f['has_negation']      = int(any(w in text.lower().split() for w in negation_words))

        features.append(f)
    return pd.DataFrame(features)

# Apply to raw text (before cleaning — we want original punctuation/caps)
# df index aligns with X_train/X_val/X_test indices
train_idx = X_train.index
val_idx   = X_val.index
test_idx  = X_test.index

hc_train = extract_handcrafted_features(raw_train)
hc_val   = extract_handcrafted_features(raw_val)
hc_test  = extract_handcrafted_features(raw_test)

print("Hand-crafted feature matrix shape:", hc_train.shape)

print("Hand-crafted feature matrix shape:", hc_train.shape)
print("\nFeature means by split:")
print(hc_train.describe().loc[['mean', 'std']].round(2))

# Combine TF-IDF with handcrafted features
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

# Scale hand-crafted features (TF-IDF is already normalized)
scaler = StandardScaler()
hc_train_scaled = scaler.fit_transform(hc_train)
hc_val_scaled   = scaler.transform(hc_val)
hc_test_scaled  = scaler.transform(hc_test)

# Combine: sparse TF-IDF + dense hand-crafted (convert to sparse for hstack)
X_train_combined = hstack([X_train_tfidf, csr_matrix(hc_train_scaled)])
X_val_combined   = hstack([X_val_tfidf,   csr_matrix(hc_val_scaled)])
X_test_combined  = hstack([X_test_tfidf,  csr_matrix(hc_test_scaled)])

print(f"Combined feature matrix shape (train): {X_train_combined.shape}")

# Sanity check
# Attach labels back to hand-crafted train features for inspection
hc_train_check = hc_train.copy()
hc_train_check['sentiment'] = y_train.values

print("Mean feature values by sentiment class:")
print(hc_train_check.groupby('sentiment').mean().round(3).T)
# 0=negative, 1=neutral, 2=positive

# Class weights
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

print("Class weights:")
for label_idx, weight in class_weight_dict.items():
    print(f"  {label_encoder.classes_[label_idx]}: {weight:.3f}")

# Logistic Regression with class weights
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)

# 5-fold CV on training data
cv_scores = cross_val_score(lr, X_train_combined, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"Logistic Regression — 5-fold CV macro F1:")
print(f"  Per fold : {[round(s, 3) for s in cv_scores]}")
print(f"  Mean ± std: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit on full train, evaluate on val
lr.fit(X_train_combined, y_train)
y_pred_lr = lr.predict(X_val_combined)

print("\nValidation set performance:")
print(classification_report(y_val, y_pred_lr, target_names=label_encoder.classes_))

# Naive Bayes 
from sklearn.naive_bayes import ComplementNB

# ComplementNB is better than MultinomialNB for imbalanced text classification
cnb = ComplementNB(alpha=0.1)

cv_scores_cnb = cross_val_score(cnb, X_train_tfidf, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f" Naive Bayes — 5-fold CV macro F1:")
print(f"  Per fold : {[round(s, 3) for s in cv_scores_cnb]}")
print(f"  Mean ± std: {cv_scores_cnb.mean():.3f} ± {cv_scores_cnb.std():.3f}")

cnb.fit(X_train_tfidf, y_train)
y_pred_cnb = cnb.predict(X_val_tfidf)

print("\nValidation set performance:")
print(classification_report(y_val, y_pred_cnb, target_names=label_encoder.classes_))

# SVM
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

svm = LinearSVC(
    C=0.1,
    class_weight='balanced',
    max_iter=2000,
    random_state=42
)

cv_scores_svm = cross_val_score(svm, X_train_combined, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"LinearSVC — 5-fold CV macro F1:")
print(f"  Per fold : {[round(s, 3) for s in cv_scores_svm]}")
print(f"  Mean ± std: {cv_scores_svm.mean():.3f} ± {cv_scores_svm.std():.3f}")

svm.fit(X_train_combined, y_train)
y_pred_svm = svm.predict(X_val_combined)

print("\nValidation set performance:")
print(classification_report(y_val, y_pred_svm, target_names=label_encoder.classes_))

# Visualization
# Confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
models = [
    ('Logistic Regression', y_pred_lr),
    ('Naive Bayes',       y_pred_cnb),
    ('LinearSVC',           y_pred_svm),
]

for ax, (name, y_pred) in zip(axes, models):
    cm = confusion_matrix(y_val, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=ax, cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar=False)
    ax.set_title(f'{name}', fontsize=12)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

plt.suptitle('Confusion matrices (% of true class)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary table
from sklearn.metrics import f1_score, accuracy_score

summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'LinearSVC'],
    'CV macro F1': [cv_scores.mean(), cv_scores_cnb.mean(), cv_scores_svm.mean()],
    'Val macro F1': [
        f1_score(y_val, y_pred_lr,  average='macro'),
        f1_score(y_val, y_pred_cnb, average='macro'),
        f1_score(y_val, y_pred_svm, average='macro'),
    ],
    'Val accuracy': [
        accuracy_score(y_val, y_pred_lr),
        accuracy_score(y_val, y_pred_cnb),
        accuracy_score(y_val, y_pred_svm),
    ]
}).round(3)

print(summary.to_string(index=False))

from sklearn.metrics import classification_report

def plot_classification_reports(models, y_true, class_names):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    axes = axes[0]  # flatten to 1D array
    
    for ax, (name, y_pred) in zip(axes, models):
        report = classification_report(y_true, y_pred, 
                                       target_names=class_names, 
                                       output_dict=True)
        metrics_df = pd.DataFrame(report).T.loc[class_names, ['precision', 'recall', 'f1-score']]
        
        im = ax.imshow(metrics_df.values, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(range(3))
        ax.set_xticklabels(['precision', 'recall', 'f1'], fontsize=11)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=11)
        ax.set_title(name, fontsize=12, pad=10)
        
        for i in range(len(class_names)):
            for j in range(3):
                val = metrics_df.values[i, j]
                text_color = 'black' if 0.35 < val < 0.85 else 'white'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=13, color=text_color, fontweight='500')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Classification report heatmaps', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('classification_reports.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_classification_reports(
    models=[
        ('Logistic Regression', y_pred_lr),
        ('Naive Bayes',       y_pred_cnb),
        ('LinearSVC',           y_pred_svm),
    ],
    y_true=y_val,
    class_names=label_encoder.classes_.tolist()
)

# Error analysis on LinearSVC
val_df = pd.DataFrame({
    'text': raw_val.values,
    'cleaned': X_val.values,
    'true': [label_encoder.classes_[i] for i in y_val],
    'predicted': [label_encoder.classes_[i] for i in y_pred_svm]
})

val_df['correct'] = val_df['true'] == val_df['predicted']
errors = val_df[~val_df['correct']].copy()

print(f"Total errors: {len(errors)} / {len(val_df)} ({len(errors)/len(val_df)*100:.1f}%)\n")

# Most common confusion pairs
confusion_pairs = errors.groupby(['true', 'predicted']).size().sort_values(ascending=False)
print("Most common mistakes:")
print(confusion_pairs)

# Sample misclassified tweets for each confusion type
print("\n── Negative predicted as Neutral ──")
sample = errors[(errors['true']=='negative') & (errors['predicted']=='neutral')]['text'].sample(4, random_state=42)
for t in sample: print(f"  • {t}")

print("\n── Neutral predicted as Negative ──")
sample = errors[(errors['true']=='neutral') & (errors['predicted']=='negative')]['text'].sample(4, random_state=42)
for t in sample: print(f"  • {t}")

print("\n── Positive predicted as Neutral ──")
sample = errors[(errors['true']=='positive') & (errors['predicted']=='neutral')]['text'].sample(4, random_state=42)
for t in sample: print(f"  • {t}")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Predict on test set
y_pred_test = svm.predict(X_test_combined)

print("=" * 50)
print("FINAL TEST SET RESULTS — LinearSVC")
print("=" * 50)
print(f"\nAccuracy  : {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Macro F1  : {f1_score(y_test, y_pred_test, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

fig, ax = plt.subplots(figsize=(7, 5))

cm = confusion_matrix(y_test, y_pred_test)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar=True)

ax.set_title('LinearSVC — Test set confusion matrix (%)', fontsize=13)
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Classification report heatmap for test set
plot_classification_reports(
    models=[('LinearSVC — test set', y_pred_test)],
    y_true=y_test,
    class_names=label_encoder.classes_.tolist()
)