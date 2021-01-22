from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
from numpy import linalg as la

def text2sents(file):
    '''sentence tokenization'''
    paras = [line.strip() for line in open(file)]
    all_sents = []
    for para in paras:
        if para:
            sents = sent_tokenize(para)
            all_sents.extend(sents)
    return all_sents

# word lemmatization
lmtzr = WordNetLemmatizer()
rule = './dict/rule.txt'
vd = {}
for line in open(rule):
    vl = line.strip().split('\t')
    for x in vl[1:]:
        if x != vl[0] and x not in vd:
            vd[x] = vl[0]

def lemmatize(word, pos):
    if pos in ['a','n','v','r','s']:
        lemma = lmtzr.lemmatize(word.lower(),pos)
    else:
        lemma = lmtzr.lemmatize(word.lower())

    if lemma == word and word in vd:
        lemma = vd[word]

    return lemma

def convert_pos(tag):
    if 'noun' in tag:
        return 'n'
    elif 'verb' in tag:
        return 'v'
    elif 'adverb' in tag:
        return 'r'
    elif 'adjective' in tag:
        return 'a'
    else:
        return tag

part = {
    'N' : 'n',
    'V' : 'v',
    'J' : 'a',
    'S' : 's',
    'R' : 'r'
}

def convert_tag(penn_tag):

    if penn_tag[0] in part.keys():
        return part[penn_tag[0]]
    else:
        # other parts of speech will be tagged as nouns
        return penn_tag

def nltk_pos(sent):
    text = word_tokenize(sent)
    wps = pos_tag(text)
    wordlist = [wp[0].lower() for wp in wps]
    poslist = [convert_tag(wp[1]) for wp in wps]
    return wordlist, poslist

def tag_and_lem(element):
    '''
    tag_and_lem() accepts a string, tokenizes, tags, converts tags,
    lemmatizes, and returns a string
    '''
    # list of tuples [('token', 'tag'), ('token2', 'tag2')...]
    sent = pos_tag(word_tokenize(element)) # must tag in context
    return ' '.join([lemmatize(sent[k][0], convert_tag(sent[k][1][0]))
                    for k in range(len(sent))])

def cosSimilarity(A,B):
    vector1 = np.mat(A)
    vector2 = np.mat(B)
    cosV12 = float(vector1*vector2.T)/(la.norm(vector1)*la.norm(vector2))
    return cosV12
