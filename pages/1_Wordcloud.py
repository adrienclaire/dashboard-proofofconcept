import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from Home import load_data

@st.cache_data
def generate_wordcloud(data):
    return WordCloud(max_words=400, width=1200, height=600, background_color="white").generate(' '.join(data))

data = load_data()

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