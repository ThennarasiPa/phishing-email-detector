import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. DATASET
# ==========================================
data = {
    'email_text': [
        "Dear customer, your bank account has been locked. Click http://secure-bank-login-update.com immediately to reset your password.",
        "URGENT: Verify your PayPal account credentials now at http://paypal-security-check.net or face suspension.",
        "Hey, are we still meeting for lunch today at 12:30 PM? Let me know.",
        "Your Amazon order #123-4567 has been shipped. Track your package here.",
        "Congratulations! You won a $1000 Walmart gift card! Click here http://free-rewards-now.biz to claim.",
        "Hi Team, please find attached the Q3 financial report for our review meeting tomorrow.",
        "ACT NOW! Your Netflix subscription failed to renew. Update billing details at http://netflix-billing-alert.org",
        "Just checking in to see how your new project is coming along. Let's catch up soon."
    ],
    'label': [1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# ==========================================
# 2. CUSTOM TRANSFORMER (FIXED & VECTOR-OPTIMIZED)
# ==========================================
class UniversalTextTransformer:
    def __init__(self, max_features=1000):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)

    def _to_series(self, X_in):
        """Convert any input to a clean Pandas Series with a reset index to guarantee alignment"""
        if isinstance(X_in, pd.DataFrame):
            series_out = X_in.iloc[:, 0]
        elif isinstance(X_in, pd.Series):
            series_out = X_in
        else:
            series_out = pd.Series(np.array(X_in).ravel())
        
        # Resetting the index enforces absolute alignment with Scikit-learn's internal matrix representations
        return series_out.astype(str).reset_index(drop=True)

    def fit(self, X, y=None):
        texts = self._to_series(X)
        self.tfidf.fit(texts)
        return self

    def transform(self, X):
        texts = self._to_series(X)

        # 1. Generate TF-IDF Sparse Matrix and construct Sparse DataFrame
        tfidf_matrix = self.tfidf.transform(texts)
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_matrix,
            columns=self.tfidf.get_feature_names_out(),
            index=texts.index
        )

        # 2. Optimized Single-Pass Feature Extraction
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        urgency_words = {'urgent', 'act now', 'locked', 'suspension', 'congratulations', 'verify'}
        
        has_url_list = []
        urgency_count_list = []
        
        for text in texts:
            text_lower = text.lower()
            # Feature 1: URL Presence
            has_url_list.append(1 if url_pattern.search(text_lower) else 0)
            # Feature 2: Keyword Counter
            urgency_count_list.append(sum(1 for word in urgency_words if word in text_lower))

        # 3. Append features safely matching the deterministic sequential index
        tfidf_df['meta__has_url'] = pd.Series(has_url_list, index=texts.index, dtype=pd.SparseDtype(int, 0))
        tfidf_df['meta__urgency_count'] = pd.Series(urgency_count_list, index=texts.index, dtype=pd.SparseDtype(int, 0))

        return tfidf_df

# ==========================================
# 3. PIPELINE
# ==========================================
model_pipeline = Pipeline([
    ('preprocessor', UniversalTextTransformer(max_features=1000)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
X = df['email_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ==========================================
# 5. TRAIN
# ==========================================
model_pipeline.fit(X_train, y_train)

# ==========================================
# 6. EVALUATION
# ==========================================
y_pred = model_pipeline.predict(X_test)

print("\n MODEL PERFORMANCE")
print("=" * 40)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ==========================================
# 7. SAVE & LOAD
# ==========================================
joblib.dump(model_pipeline, "phishing_model.pkl")
loaded_model = joblib.load("phishing_model.pkl")

# ==========================================
# 8. LIVE TESTING
# ==========================================
print("\n" + "=" * 40)
print(" LIVE EMAIL TESTING")
print("=" * 40)

# DataFrame input
sample_df = pd.DataFrame({
    'email_text': [
        "Urgent security alert: Fix your account now http://hack-link.com",
        "Hi Mom, I'm coming home."
    ]
})

# List input
sample_list = [
    "Verify your bank now http://fake-link.com",
    "Let's meet tomorrow for lunch."
]

print("\n--- DataFrame Input ---")
pred_df = loaded_model.predict(sample_df)
for email, pred in zip(sample_df['email_text'], pred_df):
    print(f"{email[:40]}... ->", "PHISHING " if pred == 1 else "SAFE ")

print("\n--- List Input ---")
pred_list = loaded_model.predict(sample_list)
for email, pred in zip(sample_list, pred_list):
    print(f"{email[:40]}... ->", "PHISHING " if pred == 1 else "SAFE ")
