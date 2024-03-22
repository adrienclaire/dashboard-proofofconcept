import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit.logger import get_logger
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import numpy as np
from streamlit.logger import get_logger
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TFDistilBertModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model


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
    sentiment_choice = st.sidebar.radio("Choisissez le sentiment des critiques:", ('Positif', 'Négatif'))

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
    class CustomBertModel(tf.keras.Model):
        def __init__(self, max_len, **kwargs):
            super(CustomBertModel, self).__init__(**kwargs)
            # Ne chargez pas le modèle DistilBERT ici. Il doit être chargé après la désérialisation.
            self.max_len = max_len
            self.dense = tf.keras.layers.Dense(512, activation='relu')
            self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

        def build(self, input_shape):
            # Chargez le modèle DistilBERT ici pour vous assurer qu'il est construit avec la forme d'entrée correcte.
            self.transformer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
            super(CustomBertModel, self).build(input_shape)

        def call(self, inputs):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = transformer_output.last_hidden_state
            cls_token = sequence_output[:, 0, :]
            x = self.dense(cls_token)
            return self.classifier(x)

        def get_config(self):
            config = super(CustomBertModel, self).get_config()
            config.update({
                "max_len": self.max_len
            })
            return config

        @classmethod
        def from_config(cls, config):
            return cls(max_len=config['max_len'])
    
    # Pour charger le modèle entier
    def load_model_and_tokenizer():
        model_path = "model/model.h5"
        tokenizer_path = "model/tokenizer"

        with custom_object_scope({'CustomBertModel': CustomBertModel}):
            loaded_model = load_model(model_path)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
        return loaded_model, tokenizer

    loaded_model, tokenizer = load_model_and_tokenizer()

    def predict_sentiment(review_text):
        encoded_review = tokenizer(review_text, max_length=293, truncation=True, padding='max_length', add_special_tokens=True, return_tensors='tf')
        input_ids = encoded_review['input_ids']
        ttention_mask = encoded_review['attention_mask']

        # Préparer les données d'entrée pour la prédiction
        input_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
        # Effectuer la prédiction
        prediction = loaded_model(input_dict)

        # Extraire la prédiction du sentiment
        sentiment = 'Positif' if tf.nn.sigmoid(prediction).numpy().flatten()[0] > 0.5 else 'Négatif'
        return sentiment

    # Interface Streamlit pour la prédiction de sentiment
    st.header('Prédiction de sentiment pour une critique')
    review_text = st.text_area("Entrez une critique de film pour prédiction de sentiment:")
    if st.button('Prédire'):
        prediction = predict_sentiment(review_text)
        st.write(f'Le sentiment prédit est: {prediction}')

    # N'oubliez pas de remplacer 'predict_sentiment' par votre fonction de prédiction réelle

if __name__ == "__main__":
    run()