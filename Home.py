import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.logger import get_logger
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


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

if __name__ == "__main__":
    run()
