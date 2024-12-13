# Sentiment Analysis on Swiggy vs Zomato Reviews


This project performs Sentiment Analysis on customer reviews of two popular food delivery platforms, Swiggy and Zomato.
The goal is to classify the sentiment of the reviews as positive or negative based on the content of the review text.

# Overview
Sentiment analysis is a natural language processing (NLP) task that involves determining whether a piece of text expresses a positive, negative, or neutral sentiment.
In this project, I use the Naive Bayes algorithm, specifically the Multinomial Naive Bayes classifier, which is widely used for text classification problems.

# Table of Contents

1. Installation

2. Usage

3. Data Preprocessing

4. Model

5. Evaluation

6. License

# Installation

To run this project locally, you need to have Python and the following libraries installed:

pip install pandas numpy scikit-learn nltk matplotlib seaborn


# Usage
1. Clone the repository
Copy code
git clone https://github.com/your-username/sentiment-analysis-swiggy-vs-zomato.git
cd sentiment-analysis-swiggy-vs-zomato
2. Data Preprocessing
The dataset consists of reviews for Swiggy and Zomato. The preprocessing steps include:

Cleaning the text: Removing special characters, converting to lowercase, and removing stopwords.
Stemming: Reducing words to their base forms using the SnowballStemmer.
Vectorization: Converting text data into numerical format using CountVectorizer.
3. Train the Model
The data is split into training and test datasets using train_test_split(). The Multinomial Naive Bayes classifier is trained on the training data.

python
Copy code
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Training the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
4. Model Evaluation
The model's performance is evaluated using accuracy and other classification metrics such as precision, recall, and F1-score.

python
Copy code
from sklearn.metrics import accuracy_score, classification_report

# Predicting sentiment for test data
y_pred = model.predict(X_test_vec)

# Evaluating the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
5. Predictions
The model predicts whether a given review is positive (1) or negative (0).

python
Copy code
# Example of a new review
new_review = ["The food was amazing!"]
new_review_vec = vectorizer.transform(new_review)

# Predicting the sentiment
prediction = model.predict(new_review_vec)
sentiment = "Positive" if prediction[0] == 1 else "Negative"
print(f"Predicted Sentiment: {sentiment}")
# Data Preprocessing
The data preprocessing steps include:

Lowercasing: Converts all text to lowercase to avoid duplication (e.g., "Good" and "good").
Stopwords Removal: Removes common words such as "the", "is", etc., which do not contribute to the sentiment.
Stemming: Reduces words to their root form (e.g., "running" becomes "run").
Vectorization: Converts the text into numerical form using CountVectorizer, which creates a bag-of-words model.

# Model
For this project, the Multinomial Naive Bayes model was chosen due to its efficiency and effectiveness in text classification tasks.

Why Naive Bayes?
Simple and Fast: Naive Bayes is computationally efficient, especially for large datasets.
Good for Text Classification: Works well with high-dimensional text data (e.g., word counts).
Assumes Feature Independence: Despite its simplicity, it often performs well on text classification tasks.

# Evaluation
The performance of the model was evaluated based on various metrics:

Accuracy: The percentage of correctly classified reviews.
Precision, Recall, F1-Score: These metrics are used to evaluate the model's performance in terms of both false positives and false negatives.
# License
This project is licensed under the MIT License - see the LICENSE file for details.
