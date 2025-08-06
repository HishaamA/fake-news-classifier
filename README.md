# Fake News Classifier
### This project is part of a series of projects in my exploration into machine learning
This project implements a machine learning model to classify news articles as either real or fake. The model is built using a Logistic Regression classifier trained on a dataset of news articles.

## Project Workflow

The project follows these steps:

1.  **Data Loading and Preprocessing:**
    *   Load the dataset (`train.csv`) into a pandas DataFrame.
    *   Handle missing values by replacing them with empty strings.
    *   Combine the 'author', 'title', and 'text' columns into a single 'content' column.

2.  **Text Preprocessing (Stemming):**
    *   Apply stemming to the 'content' column to reduce words to their root form.
    *   Remove stopwords and non-alphabetic characters during stemming.

3.  **Vectorization:**
    *   Use TF-IDF Vectorization to convert the text data into numerical feature vectors.

4.  **Splitting Data:**
    *   Split the vectorized data into training and testing sets.

5.  **Model Training:**
    *   Train a Logistic Regression model on the training data.

6.  **Model Evaluation:**
    *   Evaluate the trained model's accuracy on both the training and testing data.

7.  **Model and Vectorizer Saving:**
    *   Save the trained model and the TF-IDF vectorizer using pickle for later use.

8.  **Streamlit Application:**
    *   Create a simple Streamlit application to take user input (news title, author, and content) and use the trained model to predict if the news is real or fake.

## Project Structure

The project consists of the following files:

- `fake_news_classifier.ipynb`: A Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `model.sav`: The trained Logistic Regression model saved as a pickle file.
- `tfidf_vectorizer.sav`: The fitted TF-IDF vectorizer saved as a pickle file.
- `train.csv`: The dataset used for training the model.
- `README.md`: This file.

## Dependencies

The following libraries are required to run the notebook and the Streamlit application:

- `numpy`
- `pandas`
- `seaborn`
- `re`
- `nltk`
- `sklearn`
- `streamlit`

