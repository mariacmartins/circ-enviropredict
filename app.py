import streamlit as st
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from joblib import load
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load models
w2v_model_maize = Word2Vec.load("models/word2vec_model_maize_3mer.bin")
w2v_model_rice = Word2Vec.load("models/word2vec_model_rice_3mer.bin")

drought_model = load('models/rf_model_drought_3mer.joblib')
cold_model = load('models/rf_model_cold_3mer.joblib')

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

def circrna_to_vec(circrna_sequence, model, k=kmer_size):
    vec = np.zeros(vector_size)
    kmers = circrna_to_kmers(circrna_sequence, k=k)
    for kmer in kmers:
        if kmer in model.wv:
            vec = vec + model.wv[kmer]
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

st.markdown('##### A Machine Learning tool that allows the prediction of the possible involvement of circRNAs with abiotic stress.')

# Selecting the type of stress
stress_type = st.radio("Select the type of stress you want to predict possible involvement:", ('drought', 'cold'))

# Input seq and Submit
seq_input = st.text_area("**circRNA sequence:**", height=120, placeholder="Example of accepted format: CTCGGGCACCTCCTCCGAGACCACTGAT. At least 20 characters are required. Only one input sequence is accepted at a time.")

if st.button('Submit'):
    if is_valid_sequence(seq_input):
        seq_input = seq_input.replace("U", "T").replace('u', 't').upper()
        # Transforming input data to Word2vec format
        columns = [f'wc_3mer_{v+1}' for v in range(vector_size)]
        df_vecs = pd.DataFrame(columns=columns)
        
        if stress_type == 'drought':
            model_to_use = w2v_model_maize
        if stress_type == 'cold':  
            model_to_use = w2v_model_rice

        df_vecs = df_vecs._append(
            [dict(zip(columns, circrna_to_vec(seq_input, model_to_use)))],
            ignore_index=True
        )
        final_df = pd.concat([df_vecs], axis=1)

        # Prediction based on selected stress type
        if stress_type == 'drought':
            y_pred = drought_model.predict(final_df)
            y_pred_proba = drought_model.predict_proba(final_df)

            if y_pred[0] == 1:
                st.write(f'**Prediction result:** Possible involvement with drought stress. Probability: {y_pred_proba[:, 1][0]:.2f}.')
            else:
                st.write(f'**Prediction result:** No involvement with drought stress detected. Probability: {y_pred_proba[:, 0][0]:.2f}.')

        elif stress_type == 'cold':
            y_pred = cold_model.predict(final_df)
            y_pred_proba = cold_model.predict_proba(final_df)

            if y_pred[0] == 1:
                st.write(f'**Prediction result:** Possible involvement with cold stress. Probability: {y_pred_proba[:, 1][0]:.2f}.')
            else:
                st.write(f'**Prediction result:** No involvement with cold stress detected. Probability: {y_pred_proba[:, 0][0]:.2f}.')

    else:
        st.write('Please enter a valid circRNA sequence. Remove special characters and headers, keep only the circRNA sequence. At least 20 characters are required. Only one input sequence is accepted at a time.')