import streamlit as st
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from joblib import load
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load models
w2v_model_maize = Word2Vec.load("models/w2v_app/word2vec_model_maize_3mer.bin")
w2v_model_rice = Word2Vec.load("models/w2v_app/word2vec_model_rice_3mer.bin")

drought_model = load('models/rf_app/rf_model_drought_3mer.joblib')
cold_model = load('models/rf_app/rf_model_cold_3mer.joblib')

# Params
kmer_size = 3
vector_size = 64

# Functions
def circrna_to_kmers(circrna_sequence, k=kmer_size):
    """
    Converts the circRNA sequence into k-mers (3-mers).
    """
    return [circrna_sequence[i:i + k] for i in range(len(circrna_sequence) - k + 1)]

def circrna_to_vec(circrna_sequence, model, k=kmer_size):
    """
    Converts the circRNA sequence into a vector of features using the Word2Vec model.
    """
    vec = np.zeros(vector_size)
    kmers = circrna_to_kmers(circrna_sequence, k)
    for kmer in kmers:
        if kmer in model.wv:
            vec += model.wv[kmer]
    return vec

def is_valid_sequence(sequence):
    """
    Validates if the sequence contains only valid characters and is at least 20 nucleotides long.
    """
    sequence = sequence.replace(" ", "").replace("\n", "")
    return len(sequence) >= 20 and re.fullmatch(r'[acgtunACGTUN]+', sequence) is not None

def process_fasta_input(fasta_input):
    """
    Processes FASTA format input or sequence input.
    """
    lines = fasta_input.splitlines()
    
    if lines[0].startswith(">"):
        sequence = "".join([line.strip() for line in lines[1:]])
    else:
        sequence = "".join([line.strip() for line in lines])
    
    if not is_valid_sequence(sequence):
        return None, "Please enter a valid circRNA sequence. Remove special characters, keep only the circRNA sequence. At least 20 characters are required. Only one input sequence is accepted at a time."
    
    return sequence.upper(), None

# Streamlit UI
col1, col2 = st.columns([3, 1])

with col1:
    st.title('circ-EnviroPredict')

with col2:
    st.image('logo.png', width=80)

st.markdown('##### A Machine Learning tool that allows the prediction of the possible involvement of circRNAs with abiotic stress.')

# Select the type of stress
stress_type = st.radio("Select the type of stress you want to predict possible involvement:", ('drought', 'cold'))

# Input sequence and submit
seq_input = st.text_area("**circRNA sequence**", height=120, placeholder="Enter your circRNA sequence here. Accepted formats: Only the nucleotide sequence (for example: CTCGGGCACCTCCTCCGAGACCACTGAT...) or in FASTA format. Only one input sequence is accepted at a time.")

if st.button('Submit'):
    processed_seq, error_msg = process_fasta_input(seq_input)
    
    if error_msg:
        st.write(error_msg) 
    else:
        processed_seq = processed_seq.replace("U", "T").upper()

        columns = [f'wc_3mer_{v + 1}' for v in range(vector_size)]
        df_vecs = pd.DataFrame(columns=columns)
        
        model_to_use = w2v_model_maize if stress_type == 'drought' else w2v_model_rice

        df_vecs = df_vecs._append(
            [dict(zip(columns, circrna_to_vec(processed_seq, model_to_use)))],
            ignore_index=True
        )
        final_df = pd.concat([df_vecs], axis=1)

        if stress_type == 'drought':
            y_pred = drought_model.predict(final_df)
            y_pred_proba = drought_model.predict_proba(final_df)
            if y_pred[0] == 1:
                st.write(f'**Prediction result:** Possible involvement with drought stress.')
                st.write(f'**Probability of drought stress involvement:** {y_pred_proba[:, 1][0]:.2f}.')
            else:
                st.write(f'**Prediction result:** No involvement with drought stress detected.')
                st.write(f'**Probability of no involvement with stress condition:** {y_pred_proba[:, 0][0]:.2f}.')

        elif stress_type == 'cold':
            y_pred = cold_model.predict(final_df)
            y_pred_proba = cold_model.predict_proba(final_df)
            if y_pred[0] == 1:
                st.write(f'**Prediction result:** Possible involvement with cold stress.')
                st.write(f'**Probability of cold stress involvement:** {y_pred_proba[:, 1][0]:.2f}.')
            else:
                st.write(f'**Prediction result:** No involvement with cold stress detected.')
                st.write(f'**Probability of no involvement with stress condition:** {y_pred_proba[:, 0][0]:.2f}.')
