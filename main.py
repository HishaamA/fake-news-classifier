import re
import streamlit as st
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

model = pickle.load(open('model.sav', 'rb'))
port_stem = PorterStemmer()
def stemmer(content):
    stemmed_content = re.sub('[^a-zA-Z]','',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content_list = []
    for word in stemmed_content:
        if word not in stopwords.words('english'):
            stemmed_Word = port_stem.stem(word)
            stemmed_content_list.append(stemmed_Word)
    stemmed_content = ' '.join(stemmed_content_list)
    return stemmed_content

vectorizer = pickle.load(open('tfidf_vectorizer.sav','rb'))
def predict_fake_news(text):
    stemmed_text = stemmer(text)
    vectorized_text = vectorizer.transform([stemmed_text])
    prediction = model.predict(vectorized_text)

    if prediction[0] == 0:
        return "This is likely not fake news"
    else:
        return "This is likely fake news"

def main():
    st.title("Fake News Predictor - Machine Learning")
    title = st.text_input("Enter News Title and Author")
    content = st.text_input("Enter News content")

    output = ''

    if st.button('Predict Fake/Real'):
        output = predict_fake_news(title+content)
        st.success(output)

if __name__ == '__main__':
    main()


