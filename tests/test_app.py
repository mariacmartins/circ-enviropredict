import unittest
from gensim.models import Word2Vec
from app import is_valid_sequence, circrna_to_kmers, circrna_to_vec, process_fasta_input  

class TestSequenceFunctions(unittest.TestCase):
    def test_is_valid_sequence(self):
        # Valid sequence
        valid_sequence = "ACCACTTTTGGACCAGAACTTGTACTACATAGGAATGAAAGCCACGTCAGCAACGTATAGTTCTGCCTTTGTTAATGCACTTCCCGCCATTACCTTCATAATGGCTGTCATTTTCAGGTAAGTTTGCTTCTAATTACCAAAGACCAATCATATTAATCATTCTCTATATTTTTTACGTCATAATTTTATAGTTATTAATTCAAAAATTGCTTAAATTTTATTAGGATAGAAACTGTAAATTTGAAGAAGACACGAAGTCTTGCAAAAGTGATTGGAACAGCAATAACTGTGGGAGGAGCAATGGTTATGACGTTGTACAAAGGTCCAGCCATTGAGCTCTTTAAGACTGCTCATAGCTCTTTACACGGCGGCTCCTCGGGCACC"
        self.assertTrue(is_valid_sequence(valid_sequence))
        
        # Invalid sequence
        invalid_sequence_short = "ATG"
        self.assertFalse(is_valid_sequence(invalid_sequence_short))
        
        # Invalid sequence
        invalid_sequence_invalid_char = "ATGCXTAC"
        self.assertFalse(is_valid_sequence(invalid_sequence_invalid_char))

    def test_circrna_to_kmers(self):
      sequence = "ACCACTTTTGGACCAGAACTTGTACTACATAGGAATGAAAGCCACGTCAGCAACGTATAGTTCTGCCTTTGTTAATGCACTTCCCGCCATTACCTTCATAATGGCTGTCATTTTCAGGTAAGTTTGCTTCTAATTACCAAAGACCAATCATATTAATCATTCTCTATATTTTTTACGTCATAATTTTATAGTTATTAATTCAAAAATTGCTTAAATTTTATTAGGATAGAAACTGTAAATTTGAAGAAGACACGAAGTCTTGCAAAAGTGATTGGAACAGCAATAACTGTGGGAGGAGCAATGGTTATGACGTTGTACAAAGGTCCAGCCATTGAGCTCTTTAAGACTGCTCATAGCTCTTTACACGGCGGCTCCTCGGGCACC"
      k = 3
      expected_kmers_count = len(sequence) - k + 1

      generated_kmers = circrna_to_kmers(sequence)
      
      self.assertEqual(len(generated_kmers), expected_kmers_count, f"Expected {expected_kmers_count} k-mers, but got {len(generated_kmers)}.")
      self.assertTrue(all(len(kmer) == k for kmer in generated_kmers), f"All k-mers should have length {k}.")


    def test_circrna_to_vec(self):
        w2v_model = Word2Vec.load("models/word2vec_model_maize_3mer.bin")  

        sequence = "ACCACTTTTGGACCAGAACTTGTACTACATAGGAATGAAAGCCACGTCAGCAACGTATAGTTCTGCCTTTGTTAATGCACTTCCCGCCATTACCTTCATAATGGCTGTCATTTTCAGGTAAGTTTGCTTCTAATTACCAAAGACCAATCATATTAATCATTCTCTATATTTTTTACGTCATAATTTTATAGTTATTAATTCAAAAATTGCTTAAATTTTATTAGGATAGAAACTGTAAATTTGAAGAAGACACGAAGTCTTGCAAAAGTGATTGGAACAGCAATAACTGTGGGAGGAGCAATGGTTATGACGTTGTACAAAGGTCCAGCCATTGAGCTCTTTAAGACTGCTCATAGCTCTTTACACGGCGGCTCCTCGGGCACC"
        
        result = circrna_to_vec(sequence, w2v_model)
        
        # Check if the vector output is 64
        self.assertEqual(len(result), 64)

    def test_process_fasta_input(self):
        
        fasta_input_invalid = ">seq1\nATGCGTACGTAGCTAGCTAGX"
        sequence, error_msg = process_fasta_input(fasta_input_invalid)
        self.assertIsNone(sequence)
        self.assertEqual(error_msg, "Please enter a valid circRNA sequence. Remove special characters, keep only the circRNA sequence. At least 20 characters are required. Only one input sequence is accepted at a time.")

if __name__ == '__main__':
    unittest.main()
