from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
import json
import pickle
import os
from tqdm import tqdm

app = Flask(__name__)

# Configuration
PREDEFINED_JSON_PATH = "/Users/dima/Downloads/Electronics.json"
MODEL_FOLDER = "model_files"  # Folder to store model files
MODEL_PATH = os.path.join(MODEL_FOLDER, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_FOLDER, "vectorizer.pkl")
MAX_SAMPLES = 10000000  # Set to None to use all data

# Create model folder if it doesn't exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

def load_and_balance_data(max_samples=None, debug=False):
    try:
        # Load raw data with progress bar
        data = []
        with open(PREDEFINED_JSON_PATH, 'r') as f:
            for i, line in tqdm(enumerate(f), desc="Loading JSON lines", unit=" lines"):
                if max_samples is not None and i >= max_samples:
                    break
                data.append(json.loads(line))
        
        # Process into DataFrame with progress bar
        reviews = []
        sentiments = []

        for item in tqdm(data, desc="Processing reviews", unit=" reviews"):
            text = item.get('reviewText', '') or item.get('summary', '')
            if text.strip():
                rating = item.get('overall', 3)
                sentiment = 'positive' if rating >= 4 else 'neutral' if rating == 3 else 'negative'
                reviews.append(text)
                sentiments.append(sentiment)
        
        df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
        
        # Debug output before balancing
        if debug:
            print("\n=== Before Balancing ===")
            print(df['sentiment'].value_counts())
        
        # Balance the dataset
        df_balanced = balance_dataset(df, target_samples=None)
        
        # Debug output after balancing
        if debug:
            print("\n=== After Balancing ===")
            print(df_balanced['sentiment'].value_counts())
        
        return df_balanced
    
    except FileNotFoundError:
        print(f"Warning: {PREDEFINED_JSON_PATH} not found. Using fallback data.")
        return pd.DataFrame({
            'review': ["Great product!", "Terrible experience.", "It was okay, nothing special."],
            'sentiment': ['positive', 'negative', 'neutral']
        })

def balance_dataset(df, target_samples=None):
    """Downsample majority classes to balance the dataset"""
    # If target_samples not specified, use the size of smallest class
    if target_samples is None:
        target_samples = df['sentiment'].value_counts().min()
    
    # Group by sentiment
    groups = {sentiment: group for sentiment, group in df.groupby('sentiment')}
    
    # Downsample each group to target_samples
    balanced_groups = []
    for sentiment, group in tqdm(groups.items(), desc="Balancing dataset", unit=" classes"):
        if len(group) > target_samples:
            balanced_group = resample(group,
                                   replace=False,
                                   n_samples=target_samples,
                                   random_state=42)
        else:
            balanced_group = group
        balanced_groups.append(balanced_group)
    
    # Combine balanced groups
    return pd.concat(balanced_groups).sample(frac=1, random_state=42).reset_index(drop=True)

def save_model_and_vectorizer(model, vectorizer):
    """Save the model and vectorizer to disk"""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_model_and_vectorizer():
    """Load the model and vectorizer from disk"""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Check if model and vectorizer exist
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Loading existing model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer()
    
    # Load a small sample of data just for the default view (not for training)
    df = load_and_balance_data(max_samples=1000, debug=False)
else:
    print("Training new model...")
    # Load and balance data
    df = load_and_balance_data(max_samples=MAX_SAMPLES, debug=True)
    
    # Train the model with progress indication
    print("\nVectorizing text data...")
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(tqdm(df['review'], desc="Vectorizing", unit=" reviews"))
    
    print("\nTraining model...")
    model = MultinomialNB()
    model.fit(X_bow, df['sentiment'])
    
    # Save the model and vectorizer
    save_model_and_vectorizer(model, vectorizer)

# Precompute results for the default view (using only a small sample)
sample_df = df.sample(min(100, len(df)), random_state=42) if len(df) > 100 else df
results_df = pd.DataFrame({
    'Review': sample_df['review'],
    'Predicted Sentiment': model.predict(vectorizer.transform(sample_df['review'])),
    'Actual Sentiment': sample_df['sentiment']
})

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return jsonify(results_df.to_dict(orient='records'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        if file.filename.endswith('.csv'):
            df_new = pd.read_csv(file.stream)
            if 'review' not in df_new.columns:
                return jsonify({'error': 'CSV must contain "review" column'}), 400
            reviews = df_new['review']
            actuals = df_new.get('sentiment', ['unknown'] * len(df_new))
        
        elif file.filename.endswith('.txt'):
            reviews = pd.Series([line.decode('utf-8').strip() for line in file.stream.readlines()])
            actuals = ['unknown'] * len(reviews)
        
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Add progress bar for prediction
        predictions = []
        for review in tqdm(reviews, desc="Predicting sentiments", unit=" reviews"):
            prediction = model.predict(vectorizer.transform([review]))[0]
            predictions.append(prediction)
        
        return jsonify([{
            'Review': review,
            'Predicted Sentiment': pred,
            'Actual Sentiment': actual
        } for review, pred, actual in zip(reviews, predictions, actuals)])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)