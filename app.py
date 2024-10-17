import streamlit as st
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from joblib import load
import re

# Load models
w2v_model = Word2Vec.load("models/word2vec_model_maize.bin")
drought_model = load('models/rf_model_drought.joblib')

# Params
kmer_size = 3
vector_size = 64

# Functions
def circrna_to_kmers(circrna_sequence, k=kmer_size):
    kmers = []
    for i in range(0, len(circrna_sequence) - k + 1):
        kmer = circrna_sequence[i:i+k]
        kmers.append(kmer)
    return kmers

def circrna_to_vec(circrna_sequence, k=kmer_size):
    vec = np.zeros(vector_size)
    kmers = circrna_to_kmers(circrna_sequence, k=k)
    for kmer in kmers:
        if kmer in w2v_model.wv:
            vec = vec + w2v_model.wv[kmer]
    return vec

def is_valid_sequence(sequence):
    sequence = sequence.replace(" ", "").replace("\n", "")

    return len(sequence) >= 20 and re.fullmatch(r'[acgtunACGTUN]+', sequence) is not None

# User-Streamlit interface
col1, col2 = st.columns([3, 1])  

with col1:
  st.title('circ-EnviroPredict')
    
with col2:
  st.image('logo.png', width=80)  

st.subheader('A Machine Learning tool that allows the prediction of the possible involvement of circRNAs with abiotic stress.')
st.write('*For now, we only have predictive models to evaluate the possible involvement of circRNAs in drought stress. But in the future, we intend to add more conditions.*')


seq_input = st.text_area("**circRNA sequence:**", height=150, placeholder="Example of accepted format: CTCGGGCACCTCCTCCGAGACCACTGAT. At least 20 characters are required.")

if st.button('Submit'):
    if is_valid_sequence(seq_input):
        # Transforming input data to Word2vec format
        columns = [f'wc_3mer_{v+1}' for v in range(vector_size)]
        df_vecs = pd.DataFrame(columns=columns)
        df_vecs = df_vecs._append(
            [dict(zip(columns, circrna_to_vec(seq_input)))],
            ignore_index=True
        )
        final_df = pd.concat([df_vecs], axis=1)

        # Prediction
        y_pred = drought_model.predict(final_df)
        y_pred_proba = drought_model.predict_proba(final_df)

        if y_pred[0] == 1:
            st.write(f'**Prediction result:** Possible involvement with drought stress. Probability: {y_pred_proba[:, 1]}.')
        else:
            st.write(f'**Prediction result:** No involvement with drought stress detected. Probability: {y_pred_proba[:, 0]}.')

    else:
        st.write('Please enter a valid circRNA sequence.')