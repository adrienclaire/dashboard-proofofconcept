import streamlit as st
from Home import get_ngrams, load_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = data = load_data()

# Select option for sentiment
sentiment_choice = st.sidebar.radio("Choisissez le sentiment des critiques pour le WordCloud :", ('Positif', 'Négatif'))

n_gram_choice = st.sidebar.select_slider('Choisissez le nombre de mots pour l\'analyse n-gramme :', options=[1, 2, 3, 4, 5])

positive_data = data[data.sentiment == 1]['review']
negative_data = data[data.sentiment == 0]['review']

if sentiment_choice == 'Positif':
    st.subheader(f'Analyse {n_gram_choice}-gramme pour les avis positifs')
    fig, ax = plt.subplots()
    n_gram_data = get_ngrams(positive_data, 20, n_gram_choice)
    n_gram_data = dict(n_gram_data)
    temp = pd.DataFrame(list(n_gram_data.items()), columns = ["Common_words", 'Count'])
    sns.barplot(data=temp, x="Count", y="Common_words", orient='h', ax=ax)
    ax.set_title('Positive reviews')
    st.pyplot(fig)
else:
    st.subheader(f'Analyse {n_gram_choice}-gramme pour les avis négatifs')
    fig, ax = plt.subplots()
    n_gram_data = get_ngrams(negative_data, 20, n_gram_choice)
    n_gram_data = dict(n_gram_data)
    temp = pd.DataFrame(list(n_gram_data.items()), columns = ["Common_words", 'Count'])
    sns.barplot(data=temp, x="Count", y="Common_words", orient='h', ax=ax)
    ax.set_title('Negative reviews')
    st.pyplot(fig)