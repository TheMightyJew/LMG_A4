"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""
import copy
from math import log, isfinite
from collections import Counter

import sys, os, time, platform, nltk


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Steven Danishevski', 'id': '209202126', 'email': 'stiven@post.bgu.ac.il'}


def read_tagged_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def read_tagged_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_tagged_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_tagged_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"




# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transisions probabilities
B = {}  # emmissions probabilities
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}

def learn(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
     and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
      tagged_sentences: a list of tagged sentences, each tagged sentence is a
       list of pairs (w,t), as retunred by read_tagged_corpus().

   Returns:
      [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
  """
    for sentence in tagged_sentences:
        last_tag = START
        for pair in sentence:
            word, tag = pair

            if tag not in allTagCounts.keys():
                allTagCounts[tag] = 0
            allTagCounts[tag] += 1

            if word not in perWordTagCounts.keys():
                perWordTagCounts[word] = {}
            if tag not in perWordTagCounts[word].keys():
                perWordTagCounts[word][tag] = 0
            perWordTagCounts[word][tag] += 1

            if last_tag not in transitionCounts.keys():
                transitionCounts[last_tag] = {}
            if tag not in transitionCounts[last_tag].keys():
                transitionCounts[last_tag][tag] = 0
            transitionCounts[last_tag][tag] += 1

            if last_tag not in emissionCounts.keys():
                emissionCounts[last_tag] = {}
            if word not in emissionCounts[last_tag].keys():
                emissionCounts[last_tag][word] = 0
            emissionCounts[last_tag][word] += 1

            last_tag = tag

        if last_tag not in transitionCounts.keys():
            transitionCounts[last_tag] = {}
        if END not in transitionCounts[last_tag].keys():
            transitionCounts[last_tag][END] = 0
        transitionCounts[last_tag][END] += 1

        if last_tag not in emissionCounts.keys():
            emissionCounts[last_tag] = {}
        if END not in emissionCounts[last_tag].keys():
            emissionCounts[last_tag][END] = 0
        emissionCounts[last_tag][END] += 1

    for tag in transitionCounts.keys():
        A[tag] = {}
        transition_total = sum(transitionCounts[tag].values()) + len(transitionCounts[tag].keys())
        for next_tag in transitionCounts[tag].keys():
            distribution = (transitionCounts[tag][next_tag]+1)/transition_total
            A[tag][next_tag] = log(distribution, 2)
        for next_tag in (list(allTagCounts.keys())+[END]):
            if next_tag not in transitionCounts[tag].keys():
                A[tag][next_tag] = log(1/transition_total, 2)
    for tag in emissionCounts.keys():
        B[tag] = {}
        emission_total = sum(emissionCounts[tag].values()) + len(emissionCounts[tag].keys())
        for next_word in emissionCounts[tag].keys():
            distribution = (emissionCounts[tag][next_word]+1)/emission_total
            B[tag][next_word] = log(distribution, 2)
        B[tag][UNK] = log(1/emission_total, 2)
    res = [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]
    return res


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn()
        allTagCounts (Counter): tag counts, as specified in learn()

    Return:
        list: list of pairs

    """
    max_tag = None
    max_frequency = 0
    for tag in allTagCounts.keys():
        frequency = allTagCounts[tag]
        if frequency > max_frequency:
            max_frequency = frequency
            max_tag = tag
    most_frequent_tag = max_tag
    res = []
    for token in sentence:
        if token in perWordTagCounts.keys():
            max_tag = None
            max_frequency = 0
            for tag in perWordTagCounts[token].keys():
                frequency = perWordTagCounts[token][tag]
                if frequency > max_frequency:
                    max_frequency = frequency
                    max_tag = tag
            res.append((token, max_tag))
        else:
            res.append((token, most_frequent_tag))
    return res


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """
    tokens = copy.deepcopy(sentence)
    res = []
    tags = retrace(viterbi(tokens, A, B)[1])
    for index in range(len(sentence)):
        res.append((sentence[index], tags[index]))
    return res

def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probabilityof the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

    """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END
    sentence.insert(0, START)
    sentence.insert(len(sentence), END)
    tokens = sentence
    matrix = []
    new_col = []
    new_col.append((START, None, 0))
    matrix.append(new_col)
    for index in range(1, len(tokens)):
        new_col = []
        if tokens[index] in perWordTagCounts.keys():
            tag_list = perWordTagCounts[tokens[index]].keys()
        elif tokens[index] == END:
            tag_list = [END]
        else:
            tag_list = allTagCounts.keys()
        for tag in tag_list:
            new_col.append(predict_next_best(tokens[index], tag, matrix[index-1]))
        matrix.append(new_col)
    best_END_item = None, None, None
    for item in matrix[-1]:
        if best_END_item[2] is None or item[2] > best_END_item[2]:
            best_END_item = item
    return best_END_item


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    res = []
    t, r, p = end_item
    while r is not None:
        res.insert(0, t)
        t, r, p = r
    return res

# a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """Returns a new item (tupple)
    """
    next_best = None, None ,None
    for predecessor in predecessor_list:
        prob = predecessor[2]
        previous_tag = predecessor[0]
        if previous_tag in B.keys():
            if word in B[previous_tag].keys():
                prob += B[previous_tag][word]
            else:
                prob += B[previous_tag][UNK]
        prob += A[previous_tag][tag]
        if next_best[2] is None or prob > next_best[2]:
            next_best = tag, predecessor, prob
    return next_best


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): tthe HMM emmission probabilities.
     """
    p = 0  # joint log prob. of words and tags
    tokens = copy.deepcopy(sentence)
    tokens.insert(0, (START, START))
    tokens.insert(len(tokens), (END, END))
    for index in range(1, len(tokens)-1):
        previous_tag = tokens[index-1][1]
        if previous_tag in B.keys():
            word = tokens[index][0]
            if word in B[previous_tag].keys():
                p += B[previous_tag][word]
            else:
                p += B[previous_tag][UNK]
        p += A[previous_tag][tokens[index][1]]
    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)
    correct = 0
    correctOOV = 0
    OOV = 0
    for index in range(len(gold_sentence)):
        if gold_sentence[index][1] == pred_sentence[index][1]:
            correct_tag = 1
        else:
            correct_tag = 0
        if gold_sentence[index][0] in perWordTagCounts.keys():
            correct += correct_tag
        else:
            correctOOV += correct_tag
            OOV += 1
    return correct, correctOOV, OOV

def test():
    tagged_sentences = read_tagged_corpus('en-ud-train.upos.tsv')
    learn(tagged_sentences)
    sentence = "American forces killed Shaikh"
    a = hmm_tag_sentence(sentence.split(' '), A, B)
    b = joint_prob(a, A, B)
    d = copy.deepcopy(a)
    a.append(('aaa', 'b'))
    a.append(('bbb', 'b'))
    d.append(('aaa', 'a'))
    d.append(('bbb', 'b'))
    c = count_correct(a, d)
    debug_point = None
