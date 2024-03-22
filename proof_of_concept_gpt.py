import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit.logger import get_logger
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import numpy as np
from streamlit.logger import get_logger
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from transformers import pipeline


# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


LOGGER = get_logger(__name__)

# Chargement des données
@st.cache_data  # allow_output_mutation=True is needed for mutable outputs
def load_data():
    data = pd.read_csv('data/IMDB_clean.csv')
    return data
    
@st.cache_data
def generate_wordcloud(data):
    return WordCloud(max_words=2000, width=1200, height=600, background_color="white").generate(' '.join(data))

@st.cache_data
def get_ngrams(review, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(review)
    bag_of_words = vec.transform(review)
    sum_words = bag_of_words.sum(axis=0)
    sum_words = np.array(sum_words)[0].tolist()
    words_freq = [(word, sum_words[idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def run():
    st.set_page_config(
        page_title="IMDB Sentiment Analyse",
        page_icon="🎬",
    )
    
    data = load_data()
    
    #Titre du dashboard
    st.title('Dashboard d\'analyse des critiques de films')

    # Analyse exploratoire des données
    st.header('Analyse exploratoire des données')

    # Longueur des avis
    data['review_length'] = data['review'].apply(len)
    st.subheader('Distribution de la longueur des critiques')
    num_bins = st.sidebar.slider('Nombre de bins pour l\'histogramme', min_value=10, max_value=100, value=50)
    fig, ax = plt.subplots()
    ax.hist(data['review_length'], bins=num_bins)
    st.pyplot(fig)

    # WordCloud
    st.subheader('WordCloud des mots les plus fréquents dans les critiques')
    try:
        #words = ' '.join(data['review'].dropna())  # Assurez-vous que les données ne contiennent pas de NaN
        wordcloud = generate_wordcloud(data['review'])
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')  # Utiliser 'bilinear' pour de meilleures performances
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération du WordCloud : {e}")

    # Vous pouvez ajouter ici d'autres analyses et visualisations...
    # Select option for sentiment
    sentiment_choice = st.sidebar.radio("Choisissez le sentiment des critiques pour le WordCloud :", ('Positif', 'Négatif'))

    if sentiment_choice == 'Positif':
      # Word cloud for positive reviews
        st.subheader('WordCloud des avis positifs')
        positive_data = data[data.sentiment == 1]['review']
        #positive_data_string = ' '.join(positive_data)
        wordcloud = generate_wordcloud(positive_data)
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word cloud for positive reviews', fontsize=20)
        st.pyplot(plt)

        # Histogram for the number of characters in positive reviews
        st.subheader('Nombre de caractères dans les avis positifs')
        fig, ax = plt.subplots()
        text_len = positive_data.str.len()
        ax.hist(text_len, color='green')
        ax.set_title('Positive Reviews')
        ax.set_xlabel('Number of Characters')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        # Word cloud for negative reviews
        st.subheader('WordCloud des avis négatifs')
        negative_data = data[data.sentiment == 0]['review']
        #negative_data_string = ' '.join(negative_data)
        wordcloud = generate_wordcloud(negative_data)
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word cloud for negative reviews', fontsize=20)
        st.pyplot(plt)

        # Histogram for the number of characters in negative reviews
        st.subheader('Nombre de caractères dans les avis négatifs')
        fig, ax = plt.subplots()
        text_len = negative_data.str.len()
        ax.hist(text_len, color='red')
        ax.set_title('Negative Reviews')
        ax.set_xlabel('Number of Characters')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
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

    # Section pour la prédiction de sentiments
    # Vous devrez intégrer votre modèle de prédiction ici
    st.header('Prédiction de sentiment pour une critique')

    form = st.form(key='sentiment-form')
    user_input = form.text_area('Enter your text')
    submit = form.form_submit_button('Submit')

    if submit:
        classifier = pipeline("sentiment-analysis")
        result = classifier(user_input)[0]
        label = result['label']
        score = result['score']

        if label == 'POSITIVE':
            st.success(f'{label} sentiment (score: {score})')
        else:
            st.error(f'{label} sentiment (score: {score})')

if __name__ == "__main__":
    run()
