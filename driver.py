from collections import Counter
from math import log, isfinite

import sys, os, time, platform, nltk
import tagger


train_tagged_sentences = tagger.read_tagged_corpus('en-ud-train.upos.tsv')
test_tagged_sentences = tagger.read_tagged_corpus('en-ud-test.upos.tsv.txt')

allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn(train_tagged_sentences)

baseline_correct = 0
baseline_correctOOV = 0
baseline_OOV = 0

for tagged_sentence in test_tagged_sentences:
    sentence = [t[0] for t in tagged_sentence]
    pred = tagger.baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts)
    sentence_correct, sentence_correctOOV, sentence_OOV = tagger.count_correct(tagged_sentence, pred)

    baseline_correct += sentence_correct
    baseline_correctOOV += sentence_correctOOV
    baseline_OOV += sentence_OOV

hmm_correct = 0
hmm_correctOOV = 0
hmm_OOV = 0

for tagged_sentence in test_tagged_sentences:
    sentence = [t[0] for t in tagged_sentence]
    pred = tagger.hmm_tag_sentence(sentence, A, B)
    sentence_correct, sentence_correctOOV, sentence_OOV = tagger.count_correct(tagged_sentence, pred)

    hmm_correct += sentence_correct
    hmm_correctOOV += sentence_correctOOV
    hmm_OOV += sentence_OOV
