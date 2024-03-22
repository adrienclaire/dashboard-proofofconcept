import streamlit as st
from Home import load_data
import matplotlib.pyplot as plt

data = load_data()

# Select option for sentiment
sentiment_choice = st.sidebar.radio("Choisissez le sentiment des critiques pour le WordCloud :", ('Positif', 'Négatif'))

if sentiment_choice == 'Positif':
    # Histogram for the number of characters in positive reviews
        st.subheader('Nombre de caractères dans les avis positifs')
        positive_data = data[data.sentiment == 1]['review']
        fig, ax = plt.subplots()
        text_len = positive_data.str.len()
        ax.hist(text_len, color='green')
        ax.set_title('Positive Reviews')
        ax.set_xlabel('Number of Characters')
        ax.set_ylabel('Count')
        st.pyplot(fig)
else:
    # Histogram for the number of characters in negative reviews
    st.subheader('Nombre de caractères dans les avis négatifs')
    negative_data = data[data.sentiment == 0]['review']
    fig, ax = plt.subplots()
    text_len = negative_data.str.len()
    ax.hist(text_len, color='red')
    ax.set_title('Negative Reviews')
    ax.set_xlabel('Number of Characters')
    ax.set_ylabel('Count')
    st.pyplot(fig)