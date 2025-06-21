# 1. Data Preparation
import pandas as pd
import numpy as np
import re
import nltk
import spacy

from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_sm')

file_path = "C:\\Users\\INSPIRON\\Desktop\\AI\\ai_dev_assignment_tickets_complex_1000.xls"
df = pd.read_excel(file_path)

print(df.isnull().sum())

# Drop rows with missing labels
df.dropna(subset=['ticket_text', 'issue_type', 'urgency_level'], inplace=True)

# Add a binary feature for urgency keywords
df['contains_urgent'] = df['ticket_text'].str.contains(r'\burgent|asap|immediately|critical|soon\b', case=False).astype(int)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return " ".join(lemmatized)

tqdm.pandas()
df['cleaned_text'] = df['ticket_text'].progress_apply(preprocess_text)

print(df.head())
print(df.isnull().sum())   # for checking missing values

# 2. Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler

# used TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])

print(tfidf_matrix.shape)   # Check the Shape of the Vector Matrix

def extract_features(text):
    words = text.split()
    word_lengths = [len(w) for w in words]
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    doc = nlp(text)

    return pd.Series({
        'ticket_length': len(words),
        'avg_word_length': np.mean(word_lengths) if words else 0,
        'sentiment_score': polarity,
        'subjectivity_score': blob.sentiment.subjectivity,
        'num_exclamations': text.count('!'),
        'uppercase_words': sum(1 for w in text.split() if w.isupper()),
        'question_marks': text.count('?'),
        'contains_time_sensitive_words': int(bool(re.search(r'\b(deadline|today|now|soon|immediately|minutes|hours|asap)\b', text.lower()))),
        'is_negative_sentiment': int(polarity < -0.2),
        'is_positive_sentiment': int(polarity > 0.2),
        'num_entities': len([ent for ent in doc.ents]),
        'verb_ratio': len([t for t in doc if t.pos_ == "VERB"]) / len(doc) if len(doc) > 0 else 0
    })

custom_features = df['ticket_text'].apply(extract_features)
custom_features['contains_urgent'] = df['contains_urgent']

# Scale custom features
scaler = StandardScaler()
scaled_custom = scaler.fit_transform(custom_features)

from scipy.sparse import hstack
X = hstack([tfidf_matrix, scaled_custom])

# 3. Multi-Task Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode labels
le_issue = LabelEncoder()
le_urgency = LabelEncoder()

y_issue = le_issue.fit_transform(df['issue_type'])
y_urgency = le_urgency.fit_transform(df['urgency_level'])

# For issue type classifier
X_train_issue, X_test_issue, y_train_issue, y_test_issue = train_test_split(X, y_issue, test_size=0.2, stratify=y_issue, random_state=42)

# For urgency level classifier
X_train_urgency, X_test_urgency, y_train_urgency, y_test_urgency = train_test_split(X, y_urgency, test_size=0.2, stratify=y_urgency, random_state=42)

# Apply SMOTETomek to Urgency Training Set
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_urgency_resampled, y_train_urgency_resampled = smt.fit_resample(X_train_urgency, y_train_urgency)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

# Model 1: Issue Type Classifier
issue_model = LogisticRegression(max_iter=1000, class_weight='balanced')
issue_model.fit(X_train_issue, y_train_issue)

# Model 2: Urgency Level Classifier
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
model_rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', max_depth=12, random_state=42)
model_xgb = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=3,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    random_state=42
)

urgency_model = VotingClassifier(
    estimators=[('lr', model_lr), ('rf', model_rf), ('xgb', model_xgb)],
    voting='soft'
)

urgency_model.fit(X_train_urgency_resampled, y_train_urgency_resampled)

# Predictions
issue_preds = issue_model.predict(X_test_issue)
urgency_preds = urgency_model.predict(X_test_urgency)

# Reports  
print("\n Issue Type Classification Report:\n", classification_report(y_test_issue, issue_preds, target_names=le_issue.classes_))
print("\n Urgency Level Classification Report:\n", classification_report(y_test_urgency, urgency_preds, target_names=le_urgency.classes_))

# 4. Entity Extraction
# Product list
product_list = ['SmartWatch V2','UltraClean Vacuum','SoundWave 300','PhotoSnap Cam','Vision LED TV','EcoBreeze AC','RoboChef Blender','FitRun Treadmill','PowerMax Battery','ProTab X1']
product_list_lower = [p.lower() for p in product_list]

# Complaint keyword list
complaint_keywords = ['broken', 'late', 'error', 'not working', 'damaged', 'fail', 'crash', 'delay', 'issue', 'problem']

# Date pattern
date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?)\b'

def extract_entities(text):
    text_lower = text.lower()
    found_products = [prod for prod in product_list if prod.lower() in text_lower]
    found_dates = re.findall(date_pattern, text, re.IGNORECASE)
    found_complaints = [word for word in complaint_keywords if word in text_lower]
    return {
        'products': found_products,
        'dates': found_dates,
        'complaint_keywords': found_complaints
    }

# Apply entity extraction to each ticket
df['extracted_entities'] = df['ticket_text'].apply(extract_entities)

# 5. Integration
def predict_ticket_info(ticket_text):
    # Entity Extraction
    text_lower = ticket_text.lower()
    found_products = [prod for prod in product_list if prod.lower() in text_lower]
    found_dates = re.findall(date_pattern, ticket_text, re.IGNORECASE)
    found_complaints = [word for word in complaint_keywords if word in text_lower]
    
    entities = {
        'products': found_products,
        'dates': found_dates,
        'complaint_keywords': found_complaints
    }

    # Preprocessing
    def preprocess_text(text):
    
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        doc = nlp(" ".join(tokens))
        lemmatized = [token.lemma_ for token in doc]
        return " ".join(lemmatized)
    
    cleaned_text = preprocess_text(ticket_text)

    # TF-IDF and Custom Features 
    tfidf_features = tfidf_vectorizer.transform([cleaned_text])
    
    words = ticket_text.split()
    word_lengths = [len(w) for w in words]
    blob = TextBlob(ticket_text)
    doc = nlp(ticket_text)

    feature_row = pd.DataFrame([{
        'ticket_length': len(words),
        'avg_word_length': np.mean(word_lengths) if words else 0,
        'sentiment_score': blob.sentiment.polarity,
        'subjectivity_score': blob.sentiment.subjectivity,
        'num_exclamations': ticket_text.count('!'),
        'uppercase_words': sum(1 for w in words if w.isupper()),
        'question_marks': ticket_text.count('?'),
        'contains_time_sensitive_words': int(bool(re.search(r'\b(deadline|today|now|soon|immediately|minutes|hours|asap)\b', text_lower))),
        'is_negative_sentiment': int(blob.sentiment.polarity < -0.2),
        'is_positive_sentiment': int(blob.sentiment.polarity > 0.2),
        'num_entities': len([ent for ent in doc.ents]),
        'verb_ratio': len([t for t in doc if t.pos_ == "VERB"]) / len(doc) if len(doc) > 0 else 0,
        'contains_urgent': int(bool(re.search(r'\burgent|asap|immediately|critical|soon\b', text_lower)))
    }])

    scaled_features = scaler.transform(feature_row)
    final_features = hstack([tfidf_features, scaled_features])

    # Prediction 
    issue_prediction = le_issue.inverse_transform(issue_model.predict(final_features))[0]
    urgency_prediction = le_urgency.inverse_transform(urgency_model.predict(final_features))[0]

    return {
        'predicted_issue_type': issue_prediction,
        'predicted_urgency_level': urgency_prediction,
        'extracted_entities': entities
    }

sample_ticket = "The SoundWave 300 is not working since June 5th, 2024. This is urgent!"
result = predict_ticket_info(sample_ticket)

print("\nPrediction Result:\n", result)
