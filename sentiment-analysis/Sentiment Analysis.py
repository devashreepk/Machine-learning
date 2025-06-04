
# Sentiment Analysis GUI App - Complete Code

# Step 1: Import Required Libraries
import re
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import pickle
import nltk
# Ensure required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import os

# Step 2: Define Text Preprocessing Function
def preprocess_review(text):
    if pd.isnull(text):  # check for NaN
        return ""
    text = str(text)  # convert to string just in case
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text


# Step 3: Load Dataset
df = pd.read_csv('../Amazon_Reviews.csv', encoding='ISO-8859-1')

df = df.dropna(subset=['Reviews'])

# column name is 'Reviews'
df['Processed_Review'] = df['Reviews'].apply(preprocess_review)

# Step 4: Vectorize and Train Model
cv = CountVectorizer(max_features=5000, ngram_range=(1, 2)) 
X = cv.fit_transform(df['Processed_Review']).toarray()

# Dummy labels for training (you can replace this with real sentiment labels)
#y = [1 if i % 3 == 0 else (0 if i % 3 == 1 else -1) for i in range(len(X))]  # 1=Positive, 0=Neutral, -1=Negative
def map_rating_to_sentiment(rating):
    if rating >= 4:
        return 1     # Positive
    elif rating == 3:
        return 0     # Neutral
    else:
        return -1    # Negative

df['Sentiment'] = df['Rating'].apply(map_rating_to_sentiment)

y = df['Sentiment']

model = GaussianNB()
model.fit(X, y)

# Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

# Step 5: GUI Code
# Reload for GUI use
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Load dataset again for display
reviews = pd.read_csv('../Amazon_Reviews.csv', encoding='ISO-8859-1')

# GUI App
root = tk.Tk()
root.title("Sentiment Analysis")

# Load Emojis
happy_emoji = ImageTk.PhotoImage(Image.open('happy.png').resize((100, 100)))
neutral_emoji = ImageTk.PhotoImage(Image.open('neutral.png').resize((100, 100)))
sad_emoji = ImageTk.PhotoImage(Image.open('sad.png').resize((100, 100)))
emoji_map = {1: happy_emoji, 0: neutral_emoji, -1: sad_emoji}
sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}


# GUI Widgets
emoji_label = tk.Label(root)
emoji_label.grid(row=0, column=0, padx=10, pady=10)
review_label = tk.Label(root, wraplength=400, justify="center", font=("Helvetica", 12))
review_label.grid(row=1, column=0, padx=10, pady=10)

current_index = [0]

def update_feedback(index):
    review_text = reviews['Reviews'].iloc[index]
    processed_review = preprocess_review(review_text)
    X_review = vectorizer.transform([processed_review]).toarray()
    predicted_sentiment = model.predict(X_review)[0]
    emoji_image = emoji_map.get(predicted_sentiment, neutral_emoji)
    emoji_label.config(image=emoji_image)
    emoji_label.image = emoji_image
    sentiment_text = sentiment_map.get(predicted_sentiment, "Neutral")
    review_label.config(text=f"{review_text}\n\nsentiment:{sentiment_text}")
    # messagebox.showinfo("Sentiment Result", f"Review:\n{review_text}\n\nSentiment: {sentiment_text}")

def next_review():
    current_index[0] = (current_index[0] + 1) % len(reviews)
    update_feedback(current_index[0])

def prev_review():
    current_index[0] = (current_index[0] - 1) % len(reviews)
    update_feedback(current_index[0])

# Buttons
next_button = ttk.Button(root, text="Next Review", command=next_review)
next_button.grid(row=2, column=0, pady=(0, 10))
prev_button = ttk.Button(root, text="Previous Review", command=prev_review)
prev_button.grid(row=3, column=0, pady=(0, 20))

# Launch the GUI
update_feedback(current_index[0])
root.mainloop()

