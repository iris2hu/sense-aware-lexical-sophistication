import glob, math, os, csv
import numpy as np
import pickle as p

# load word frequencies and estimated sense frequencies
with open('./dict/word_freq.data', 'rb') as f:
    word_freq = p.load(f)

with open('./dict/sense_freq.data', 'rb') as f:
    sense_freq = p.load(f)

def get_lexical_sophistication_indices(file, freq_threshold=3000):
    
    '''
    Version1. return the six sense-aware lexical sophistication indices
    - Ratio_Sophis_Token
    - Ratio_SA_Sophis_Token
    - Ratio_Sophis_Type
    - Ratio_SA_Sophis_Type
    - Mean_Freq_Word
    - Mean_Freq_Sense
    '''

    wordlist = []
    for line in open(file):
        wl = line.strip().split(' ')
        if wl:
            wordlist.extend(wl)

    wordlist = [w.lower() for w in wordlist if w.split('_')[0].isalpha()]
    sophis_words, sa_sophis_words = [], []
    freq1, freq2 = [], []
    
    # consider only the sophistication words
    for w in wordlist:
        w = w.split('_')[0]
        freq = word_freq.get(w, None)
        if freq and freq < freq_threshold and len(w) > 2:
            sophis_words.append(w)
        if freq:
            freq1.append(freq)

    # integrate the sense-aware information
    for w in wordlist:
        if w in sense_freq:
            freq = sense_freq.get(w)
        else:
            freq = word_freq.get(w, None)
        if freq and freq < freq_threshold:
            sa_sophis_words.append(w)
        if freq:
            freq2.append(freq)

    
    p1 = len(sophis_words) / math.sqrt(len(wordlist))   # Ratio_Sophis_Token
    p2 = len(sa_sophis_words) / math.sqrt(len(wordlist))    # Ratio_SA_Sophis_Token
    p3 = len(set(sophis_words)) / math.sqrt(len(set(wordlist))) # Ratio_Sophis_Type
    p4 = len(set(sa_sophis_words)) / math.sqrt(len(set(wordlist)))  # Ratio_SA_Sophis_Type
    l1 = np.mean([math.log(f1) for f1 in freq1])    # Mean_Freq_Word
    l2 = np.mean([math.log(f2) for f2 in freq2])    # Mean_Freq_Sense

    return p1, p2, p3, p4, l1, l2


if __name__ == '__main__':

    out_file = 'indices.csv'
    
    files = glob.glob('./output/*.txt')

    with open(out_file, 'w') as f:

        headers = ['filename', 'Ratio_Sophis_Token', 'Ratio_SA_Sophis_Token', 'Ratio_Sophis_Type', 'Ratio_SA_Sophis_Type', 'Mean_Freq_Word', 'Mean_Freq_Sense']
        f_csv = csv.writer(f)
        f_csv.writerow(headers)

        for file in files:
            filename = os.path.split(file)[-1]
            p1, p2, p3, p4, l1, l2 = get_lexical_sophistication_indices(file)
            row = [filename, p1, p2, p3, p4, l1, l2]
            f_csv.writerow(row)
