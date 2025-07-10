import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Step 1: Load dataset
# Example: Restaurant_Reviews.tsv (Tab-separated file with 'Review' and 'Liked' columns)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Step 2: Text cleaning and preprocessing
corpus = []
ps = PorterStemmer()
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # Remove non-letters
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Step 3: Feature Extraction (Bag of Words Model)
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset['Liked']  # 1 = Positive, 0 = Negative

# Step 4: Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Train a classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
