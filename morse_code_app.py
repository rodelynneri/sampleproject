# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Morse code dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.'
}


# Generate synthetic data (5000 samples of Morse code messages)
def generate_data(num_samples=5000):
    data = []
    for _ in range(num_samples):
        random_text = ''.join(np.random.choice(list(MORSE_CODE_DICT.keys()), size=np.random.randint(5, 20)))
        morse_code = ' '.join(MORSE_CODE_DICT[char] for char in random_text)
        data.append((random_text, morse_code))
    return pd.DataFrame(data, columns=['Text', 'MorseCode'])


# Generate the dataset
df = generate_data()


# Display EDA information
st.title("EDA on Morse Code Data")
st.write("### Sample Data")
st.write(df.head())


# Check for null values
st.write("### Null Values Check")
st.write(df.isnull().sum())


# Display summary statistics
st.write("### Summary Statistics")
st.write(df.describe(include='all'))


# Exploratory Data Analysis (EDA) Plots
st.write("### Length Distribution of Text and Morse Code")
df['Text Length'] = df['Text'].apply(len)
df['Morse Length'] = df['MorseCode'].apply(len)


# Distribution plots for text and Morse code lengths
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(df['Text Length'], ax=ax[0], kde=True, color='blue')
sns.histplot(df['Morse Length'], ax=ax[1], kde=True, color='orange')
ax[0].set_title("Text Length Distribution")
ax[1].set_title("Morse Code Length Distribution")
st.pyplot(fig)


# Data preparation for modeling
st.write("### Data Preparation for Modeling")
X = df['MorseCode']
y = df['Text']


# Convert Morse code into a vector using bag-of-words (BOW) encoding
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)


# Model training (Random Forest Classifier)
st.write("### Model Training")
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)


# Model predictions
y_pred = rfc.predict(X_test)


# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)


st.write(f"### Model Evaluation Metrics")
st.write(f"**Accuracy**: {accuracy:.2f}")
st.write(f"**Precision**: {precision:.2f}")
st.write(f"**Recall**: {recall:.2f}")
st.write(f"**F1-Score**: {f1:.2f}")


# Confusion Matrix
st.write("### Confusion Matrix")
confusion_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='coolwarm', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)


# Display classification report
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))


# User input section for Morse code prediction
st.write("### Try It Yourself")
user_input = st.text_input("Enter Morse Code to Decode (e.g., .... . .-.. .-.. ---)")
if user_input:
    user_input_vectorized = vectorizer.transform([user_input])
    user_prediction = rfc.predict(user_input_vectorized)
    st.write(f"**Predicted Text**: {user_prediction[0]}")


