from Bio import SeqIO
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

kmer_size = 3
vector_size = 64

def circrna_to_kmers(circrna_sequence, k=kmer_size):
  kmers = []
  for i in range(0, len(circrna_sequence)-k+1):
    kmer = circrna_sequence[i:i+k]
    kmers.append(kmer)
  return kmers

fasta_handle = open('osaj43883_genomic_seq.txt', 'r')
fasta_parser = SeqIO.parse(fasta_handle, 'fasta')

with open('oryza_corpus.txt', 'w') as corpus_handle:
  for record in fasta_parser:
    record_kmers = circrna_to_kmers(str(record.seq))
    corpus_handle.write(' '.join(record_kmers) + '\n')

w2v_model = Word2Vec(vector_size=vector_size)
w2v_model.build_vocab(corpus_file='oryza_corpus.txt')

w2v_model.corpus_count

# Training the w2v model
w2v_model.train(corpus_file='oryza_corpus.txt', total_words=w2v_model.corpus_total_words, epochs=1)

# Save the model
w2v_model.save("word2vec_model_rice_3mer.bin")