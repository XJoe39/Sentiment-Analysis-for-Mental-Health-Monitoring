
import pandas as pd  # for data manipulation
from sklearn.model_selection import train_test_split  # for splitting the dataset
from sklearn.feature_extraction.text import CountVectorizer  # for converting texts into numerical
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.metrics import classification_report, accuracy_score  # for model evaluation
import re  # for text preprocessing

# define the dataset location
dataset_path = r'C:\Users\cdc\Desktop\Uni\Level 6\Semester A\Advanced AI\NLP\Combined Data.csv'

# loading dataset using pandas
data = pd.read_csv(dataset_path)  # Read the CSV file into a DataFrame

# drop columns
data = data.drop(columns=['Unnamed: 0'])  # Remove the index column since it's not needed

# map status labels to binary categories
data['status'] = data['status'].apply(lambda x: 'abnormal' if x.lower() in ['anxiety', 'depression', 'stress'] else 'normal')  # Convert to 'abnormal' or 'normal'

# replacing NaN or non-string values in 'statement' with an empty string
data['statement'] = data['statement'].fillna('').astype(str)  # ensure all values in "statement" are strings

# define a text preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-alphanumeric characters
    text = text.lower()  # converting texts to lowercase
    return text

# applying the text preprocessing function to the "statement" column
data['statement'] = data['statement'].apply(preprocess_text)  # clean text data

# splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['statement'], data['status'], test_size=0.2, random_state=42  # 80% training, 20% testing
)

# converting text data into words for representation
vectorizer = CountVectorizer(max_features=1000)  # create a vectorizer with maximum of 1000 features
X_train_bow = vectorizer.fit_transform(X_train)  # fit and transform training data
X_test_bow = vectorizer.transform(X_test)  # transform testing data

# training Naive Bayes classifier
nb_model = MultinomialNB()  # initialize the Naive Bayes model
nb_model.fit(X_train_bow, y_train)  # training model on the training data
nb_predictions = nb_model.predict(X_test_bow)  # predict the testing data

# training Logistic Regression classifier
lr_model = LogisticRegression(max_iter=1000, random_state=42)  # initialize the Logistic Regression model
lr_model.fit(X_train_bow, y_train)  # Training model on the training data
lr_predictions = lr_model.predict(X_test_bow)  # predict the testing data

# evaluating Naive Bayes classifier
print("Naive Bayes Classifier Results:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))  # print accuracy
print(classification_report(y_test, nb_predictions))  # print detailed metrics

# evaluating Logistic Regression classifier
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, lr_predictions))  # print accuracy
print(classification_report(y_test, lr_predictions))  # print detailed metrics
