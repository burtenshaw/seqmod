import numpy as np
import nltk
from nltk import word_tokenize

# Levenstein
from nltk.metrics import *
# Stopwords
from nltk.corpus import stopwords
# To use the stop words list:
stops = set(stopwords.words("english"))

def sentence_tokenizer(string):
    # Break up a text into sentences
    import nltk.data
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_list = sent_detector.tokenize(string.strip())
    return sentence_list

# Handling words as simple strings:

def sieve(string_list):
    # Remove extras from a list
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out

def crossBagger(X,Y):
    # Turn a two lists into one, with no extras.
    words = []
    for i in X:
        if i in Y:
            words.append(i)
    words = sieve(words)
    return words

# Handling words via their part of speach.

def posScraper(string, X):
    # Scrape all the words from a string, ascociated to one POS.
    text = nltk.pos_tag(word_tokenize(string))
    pos = []
    for i in text:
        if i[1].startswith(X):
            pos.append(i[0])
    return pos

def phraseScraper(string, POS, window):
    pos_list = posScraper(string, POS)
    list = string.split()
    pos_phrases = []
    for i,item in enumerate(list):
        if item in pos_list:
            phrase = list[i-window], list[i], list[i+window]
            pos_phrases.append(phrase)
    return pos_phrases

def nounPhrases(string):
    '''
    Breaks up a sentence into chunks of nouns.
    For example, ['The Blue School'], rather than ['the', 'blue', 'school']
    '''
    grammar = r"""
        NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
        {<NNP>+}                # chunk sequences of proper nouns
        {<NN>+}                 # chunk consecutive nouns
        """

    cp = nltk.RegexpParser(grammar)
    tagged_sent = nltk.pos_tag(string.split())
    parsed_sent = cp.parse(tagged_sent)
    for subtree in parsed_sent.subtrees():
      if subtree.label() == 'NP':
        yield ' '.join(word for word, tag in subtree.leaves())

def nounPropper(string):
    '''
    Tokenises the propper nouns.
    '''
    grammar = r"""
        NNP: {<DT|PP\$>?<JJ>*<NNP>}
        {<NNP>+}
        """
    cp = nltk.RegexpParser(grammar)
    tagged_sent = nltk.pos_tag(string.split())
    parsed_sent = cp.parse(tagged_sent)
    for subtree in parsed_sent.subtrees():
      if subtree.label() == 'NNP':
        yield ' '.join(word for word, tag in subtree.leaves())


def propperNounList(string):
    # takes in a plain string and gives back a list of the propper nouns.
    sent_tokenized = word_tokenize(string)
    pn_list = []
    for npstr in nounPropper(string):
         pn_list.append(npstr)
    return pn_list

# Handling via their characters:

def questionList(list):
    # Turn a text into a list of question pairs
    questions_answers = []
    for i in list:
        if "?" in i:
            pair = []
            pair.append(i)
            pair.append(i + 1)
            questions_answers.append(tuple(pair))
    return questions_answers


def levenstein(s1, s2):
    X = distance.edit_distance(s1, s2)
    return X

def jaccard_sim(x,y):
    """
    Compute Jaccard similarity between two bags of words
    """
    return len(set(x) & set(y)) / float(len(set(x) | set(y)))

def smith_waterman(seq1, seq2, match=3, mismatch=-1, insertion=-1, deletion=-1, normalize=1):
    '''Temporarily borrowed from stolen from http://climberg.de/page/smith-waterman-distance-for-feature-extraction-in-nlp/'''
    # switch sequences, so that seq1 is the longer sequence to search for seq2
    if len(seq2) > len(seq1): seq1, seq2 = seq2, seq1
    # create the distance matrix
    mat = np.zeros((len(seq2) + 1, len(seq1) + 1))
    # iterate over the matrix column wise
    for i in range(1, mat.shape[0]):
        # iterate over the matrix row wise
        for j in range(1, mat.shape[1]):
            # set the current matrix element with the maximum of 4 different cases
            mat[i, j] = max(
                # negative values are not allowed
                0,
                # if previous character matches increase the score by match, else decrease it by mismatch
                mat[i - 1, j - 1] + (match if seq1[j - 1] == seq2[i - 1] else mismatch),
                # one character is missing in seq2, so decrease the score by deletion
                mat[i - 1, j] + deletion,
                # one additional character is in seq2, so decrease the scare by insertion
                mat[i, j - 1] + insertion
            )
    # the maximum of mat is now the score, which is returned raw or normalized (with a range of 0-1)
    return np.max(mat) / (len(seq2) * match) if normalize else np.max(mat)

# expand into function based on synset
# positives = ['yes','ok','sure']


# there may be several references

def bleu(hypothesis, reference):
    hypothesis = word_tokenize(hypothesis)
    reference = word_tokenize(reference)
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    return BLEUscore


# hypothesis = 'It is a cat at room'
# reference = 'It is a cat inside the room'

# print(bleu(hypothesis, reference))
