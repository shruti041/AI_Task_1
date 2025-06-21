# TASK – 1
## Objective:
### Develop a machine learning pipeline that classifies customer support tickets by their issue type and urgency level, and extracts key entities (e.g., product names, dates, complaint keywords). The file (ai_dev_assignment_tickets_complex_1000 ) is provided.
# 1. Data Preparation
## a. Cleaning & Filtering
- Remove rows with missing values in essential fields.  <br>
- Normalize text (lowercasing, removing URLs, digits, and punctuation).
## b. Text Preprocessing  
Tokenize text, remove stopwords, and perform lemmatization using spaCy.  
## c. Feature Flag  
Add binary flag contains_urgent if text contains urgency-indicating words like “urgent”, “asap”, etc.
# 2. Feature Engineering
## a. TF-IDF Vectorization
Extracts n-gram (1–2) based term importance features using TF-IDF for 1500 most relevant terms.
## b. Custom Linguistic Features
1. ticket_length
   - **Description**: Total number of words in the ticket.
   - **Purpose**: Longer tickets may indicate more complex issues.
2. avg_word_length
   - **Description**: Average length of words used in the ticket.
   - **Purpose**: Indicates vocabulary complexity or formality of the text.
3. sentiment_score
   - **Description**: Sentiment polarity of the ticket using TextBlob.
   - **Range**: -1.0 (very negative) to 1.0 (very positive)
   - **Purpose**: Helps identify emotional tone—frustrated (negative) vs appreciative (positive).
4. subjectivity_score
   - **Description**: Degree of subjectivity in the ticket using TextBlob.
   - **Range**: 0.0 (very objective) to 1.0 (very subjective)
   - **Purpose**: Indicates how factual or opinion-based the ticket is.
5. num_exclamations
   - **Description**: Number of `!` characters in the text.
   - **Purpose**: High count may indicate frustration, urgency, or emphasis.
6. uppercase_words
   - **Description**: Count of fully uppercase words (e.g., “URGENT”, “NOT WORKING”).
   - **Purpose**: May highlight emphasis or urgency.
7. question_marks
   - **Description**: Number of `?` characters in the text.
   - **Purpose**: Suggests user confusion or requests for clarification.
8. contains_time_sensitive_words
   - **Description**: Binary feature (0 or 1) indicating presence of urgency-related keywords (e.g., “today”, “now”, “asap”).
   - **Purpose**: Directly flags tickets that imply urgency or time sensitivity.
9. is_negative_sentiment
   - **Description**: Set to 1 if sentiment polarity < -0.2 (strongly negative).
   - **Purpose**: Helps model detect dissatisfaction or complaints.
10. is_positive_sentiment
    - **Description**: Set to 1 if sentiment polarity > 0.2 (strongly positive).
    - **Purpose**: Captures praise or appreciative tickets.
11. num_entities
    - **Description**: Count of named entities (products, dates, locations, etc.) detected using spaCy.
    - **Purpose**: Measures how many meaningful named items are mentioned.
12. verb_ratio
    - **Description**: Ratio of verbs to total tokens in the ticket.
    - **Purpose**: Indicates action-orientation—may correlate with user intentions, complaints, or instructions.
12. contains_urgent
    - **Description**: 1 if words like "urgent", "asap", "immediately", "critical", "soon" appear in the text.
    - **Purpose**: Direct keyword-based flag for urgent tickets.
## c. Scaling
Numerical features are scaled using StandardScaler for uniformity before feeding into classifiers.
# 3. Multi-Task Learning
## a. Label Encoding
Converts categorical labels (issue_type, urgency_level) to numeric form.
## b. Train-Test Split
The data is split into training and test sets with stratified sampling to preserve label distribution.
## c. Handling Imbalance
SMOTETomek is applied to the urgency dataset to:
- Over-sample minority classes (SMOTE)
- Clean overlapping instances (Tomek links)
## d. Model Training
**Issue Type Model:** Logistic Regression (handles imbalanced data with class_weight=balanced).<br>
**Urgency Level Model:** Soft Voting Classifier using:
- Logistic Regression
- Random Forest
- XGBoost (multi-class)
# 4. Entity Extraction
Uses pattern matching and predefined lists to extract:
- **Product Names:** From a fixed list.
- **Complaint Keywords:** Keywords indicating issues (e.g., "broken", "delay").
- **Dates:** Regex patterns matching various date formats.
This helps in understanding ticket context beyond just classification.
# 5. Integration
The predict_ticket_info() function:
- Extracts entities.
- Preprocesses the text.
- Generates features (TF-IDF + custom features).
- Predicts issue_type and urgency_level using trained models.
- Returns both predictions and extracted entities.
# Key Design Choices
## 1. Multi-Task Learning Setup
- **Choice:** Separate classifiers for issue_type and urgency_level.
- **Why:** These two labels have different distributions and characteristics. Treating them separately allows optimized training and evaluation for each task.
## 2. Text Preprocessing
- **Choice:** Lowercasing, removing URLs, numbers, special characters, stopword removal, lemmatization (with spaCy).
- **Why:** Ensures consistency in text data and reduces noise. Lemmatization helps normalize word.
## 3. Binary Feature Flag (contains_urgent)
- **Choice:** Add a simple binary feature to indicate presence of urgency words.
- **Why:** Urgent tickets often contain specific keywords. This domain-specific feature boosts urgency classification performance.
## 4. TF-IDF Vectorization
- **Choice:** Use TfidfVectorizer with n-grams (1,2), max 1500 features.
- **Why:** Captures local context and common phrases like “not working”, “payment failed”, which are essential in support tickets. TF-IDF helps highlight discriminative words.
## 5. Custom Linguistic and Semantic Features
- **Choice:** Include features like sentiment score, length, punctuation, verb ratio, etc.
- **Why:** These features provide structural and emotional cues.
## 6. Combining Features
- **Choice:** Combine TF-IDF features (sparse) with scaled custom features (dense) using scipy.hstack.
- **Why:** Merges semantic (text) and syntactic/statistical features into one representation for better learning.
## 7. Class Imbalance Handling (SMOTETomek)
- **Choice:** Use SMOTETomek on urgency training set.
- **Why:** Urgency labels are typically imbalanced (few "High" urgency tickets). SMOTETomek oversamples minority and cleans ambiguous samples, improving model fairness and recall.
## 8. Model Selection
**Issue Type:** Logistic Regression with class weights.
- **Why:** Simple, interpretable, good baseline for multi-class.
**Urgency Level:** Soft Voting Classifier combining Logistic Regression, Random Forest, and XGBoost.
- **Why:** Ensemble improves robustness and captures both linear and non-linear relationships.
## 9. Entity Extraction
- **Choice:** Use regex + keyword lists to extract products, complaints, and dates.
- **Why:** Quick and interpretable method to get domain-relevant info. Helps with explainability and context enrichment.
## 10. Modular Prediction Pipeline
- **Choice:** predict_ticket_info() encapsulates preprocessing, feature extraction, prediction, and entity extraction.
- **Why:** Makes the system reusable, scalable, and easy to integrate into APIs or web interfaces.
# Model evaluation (with metrics)
Both models were evaluated using Precision, Recall, and F1-Score to ensure balanced performance across categories.
# Limitations
## Imbalanced Data:
Urgency levels are often skewed (e.g., more “Low” than “High”), which can affect model generalization despite using SMOTETomek.
## Keyword Dependence:
Features like contains_urgent or complaint_keywords rely on fixed patterns, limiting adaptability to new language or phrasing.
## Rule-Based Entity Extraction:
Product and complaint identification is not context-aware and may miss or misclassify entities not in the predefined lists.
## No Deep Contextual Understanding:
Classical models (e.g., Logistic Regression, Random Forest) can't fully capture semantic meaning compared to transformer models (like BERT).
## Multilingual Limitation:
The pipeline supports only English tickets; non-English inputs are not handled.






