"""
classify.py
"""
import datetime
import glob
import json
import os
import re
import urllib.request
import urllib.request
import urllib.request
from collections import defaultdict
from io import BytesIO
from itertools import combinations
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

neg_words = set()
pos_words = set()


def get_afinn():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    for k, v in afinn.items():
        if v > 0:
            pos_words.add(k)
        else:
            neg_words.add(k)
    return pos_words, neg_words


def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    tokenList = []
    if keep_internal_punct == False:
        return np.array(re.sub('\W', ' ', doc).lower().split())
    if keep_internal_punct == True:
        splitResult = doc.split(' ');
        for word in splitResult:
            word = re.sub("[\W]+$", '', word, flags=re.UNICODE)
            word = re.sub("^[\W]+", '', word, flags=re.UNICODE)
            if len(word) > 0:
                tokenList.append(word.lower().strip())
        return np.array(tokenList)


def token_features(tokens, feats):
    for token in tokens:
        if 'token=' + token in feats:
            feats['token=' + token] = (feats['token=' + token] + 1)
        else:
            feats['token=' + token] = 1


def token_pair_features(tokens, feats, k=3):
    for i in range(0, len(tokens)):
        if i + k <= len(tokens):
            subSet = tokens[i:i + k];
            for j in range(0, len(subSet) - 1):
                for l in range(j + 1, len(subSet)):
                    if ('token_pair=' + subSet[j] + '__' + subSet[l]) in feats:
                        feats['token_pair=' + subSet[j] + '__' + subSet[l]] = (
                            feats['token_pair=' + subSet[j] + '__' + subSet[l]] + 1)
                    else:
                        feats['token_pair=' + subSet[j] + '__' + subSet[l]] = 1


def lexicon_features(tokens, feats):
    feats['neg_words'] = 0;
    feats['pos_words'] = 0;
    for token in tokens:
        if token.lower() in [x.lower() for x in neg_words]:
            feats['neg_words'] = (feats['neg_words'] + 1)
        if token.lower() in [x.lower() for x in pos_words]:
            feats['pos_words'] = (feats['pos_words'] + 1)


def featurize(tokens, feature_fns):
    featureList = []
    for funcDef in feature_fns:
        feats = dict()
        funcDef(tokens, feats)
        for items in feats.items():
            featureList.append(items)
    return sorted(featureList, key=lambda tup: tup[0])


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    vocabulary = defaultdict(lambda: 0)
    indices = []
    data = []
    indptr = [0]
    vocabList = defaultdict(lambda: 0)
    featureList = []
    for tokens in tokens_list:
        feats = featurize(tokens, feature_fns)
        featureList.append(feats)
    if vocab == None:
        for featureElements in featureList:
            for term in featureElements:
                index = vocabulary.setdefault(term, len(vocabulary))
                if term[0] in vocabList:
                    vocabList[term[0]] += 1
                else:
                    vocabList[term[0]] = 1
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))
        finalVocabList = dict((k, v) for k, v in vocabList.items() if v >= min_freq)
    else:
        finalVocabList = vocab

    count = 0
    returnVocabList = defaultdict(lambda: 0)
    for vocabs in sorted(finalVocabList.items(), key=lambda tup: tup[0]):
        returnVocabList[vocabs[0]] = count
        count += 1

    indptr1 = [0]
    data1 = []
    indices1 = []
    for featureElements in featureList:
        for element in featureElements:
            if element[0] in finalVocabList:
                indices1.append(returnVocabList[element[0]])
                data1.append(element[1])
        indptr1.append(len(indices1))
    X = csr_matrix((data1, indices1, indptr1), dtype='int64')
    return (X, returnVocabList)


def accuracy_score(truth, predicted):
    return len(np.where(truth == predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(len(labels), k)
    accuracies = []
    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    listOfDict = []
    clf = LogisticRegression()
    for punctuation in punct_vals:
        for minFrequency in min_freqs:
            for L in range(0, len(feature_fns) + 1):
                for subset in combinations(feature_fns, L):
                    if subset:
                        X, vocabs = vectorize([tokenize(d, punctuation) for d in docs], subset, minFrequency,
                                              vocab=None)
                        accurracy = cross_validation_accuracy(clf, X, labels, 5)
                        listOfDict.append(
                            {'features': subset, 'punct': punctuation, 'accuracy': accurracy, 'min_freq': minFrequency})
    listOfDict.sort(key=lambda x: (-x['accuracy'], -x['min_freq']))
    return listOfDict


def fit_best_classifier(docs, labels, best_result):
    minFreq = best_result['min_freq']
    punct = best_result['punct']
    features = best_result['features']
    X, vocab = vectorize([tokenize(d, punct) for d in docs], features, minFreq, vocab=None)
    clf = LogisticRegression()
    clf.fit(X, labels)
    return (clf, vocab)


def removeLinks(tweet):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)


def removeHashTagsAndUsersName(linkFreeTweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", linkFreeTweet).split())

def parse_tweets(filePath):
    tweets = list()
    f = open(filePath, 'r')
    for line in f:
        if len(line) > 1:
            tweet = json.loads(line)['text']
            linkFreeTweet = removeLinks(tweet)
            hashtagFreeTweet = removeHashTagsAndUsersName(linkFreeTweet)
            tweets.append(hashtagFreeTweet)
    return tweets


def parse_test_data(best_result, vocab, testDocs):
    minFreq = best_result['min_freq']
    punct = best_result['punct']
    features = best_result['features']
    X, vocab = vectorize([tokenize(d, punct) for d in testDocs], features, minFreq, vocab)
    return testDocs, X


def downloadData():
    # The file is 78M, so this will take a while.
    url = urllib.request.urlopen('http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    # We'll focus on the smaller file that was manually labeled.
    # The larger file has 1.6M tweets "pseudo-labeled" using emoticons
    # training.1600000.processed.noemoticon.csv
    # testdata.manual.2009.06.14.csv
    return zipfile.open('testdata.manual.2009.06.14.csv')


def read_training_data(trainingData):
    tweets = pd.read_csv(trainingData,
                         header=None,
                         names=['polarity', 'id', 'date',
                                'query', 'user', 'text'])
    labels = np.array(tweets['polarity'])
    docs = tweets['text'].tolist()
    return docs, labels


def get_analyzed_result(predictions, predictedProbabilities, test_docs):
    positiveTweets = list()
    neutralTweets = list()
    negativeTweets = list()
    entries = []
    for index in range(len(predictions)):
        entry = dict()
        entry['predicted'] = predictions[index]
        probabIndex = 0
        if entry['predicted'] == 2:
            probabIndex = 1
        if entry['predicted'] == 4:
            probabIndex = 2
        entry['proba'] = predictedProbabilities[index][probabIndex]
        entry['document'] = test_docs[index]
        entries.append(entry)
    for docEntry in entries:
        if docEntry['predicted'] == 0:
            negativeTweets.append(docEntry)
        if docEntry['predicted'] == 4:
            positiveTweets.append(docEntry)
        if docEntry['predicted'] == 2:
            neutralTweets.append(docEntry)

    sortedPositiveTweets = sorted(positiveTweets, key=lambda x: -x['proba'])
    sortedNegativeTweets = sorted(negativeTweets, key=lambda x: -x['proba'])
    sortedNeutralTweets = sorted(neutralTweets, key=lambda x: -x['proba'])

    with open('classify.txt', 'w') as classifyResult:
        classifyResult.write('Classify Results: ')
        classifyResult.write('\n \n')
        classifyResult.write('Total number of tweets processed: ' + str(len(test_docs)))
        classifyResult.write('\n')
        classifyResult.write('Number of positive tweets classified : ' + str(len(positiveTweets)))
        classifyResult.write('\n')
        classifyResult.write('Positive tweet sample : ' + str(sortedPositiveTweets[0]['document']))
        classifyResult.write('\n \n')
        classifyResult.write('Number of negative tweets classified : ' + str(len(negativeTweets)))
        classifyResult.write('\n')
        classifyResult.write('Negative tweet sample : ' + str(sortedNegativeTweets[0]['document']))
        classifyResult.write('\n \n')
        classifyResult.write('Number of neutral tweets classified : ' + str(len(sortedNeutralTweets)))
        classifyResult.write('\n')
        classifyResult.write('Neutral tweet sample : ' + str(sortedNeutralTweets[0]['document']))


def main():
    startTime = datetime.datetime.now()

    get_afinn()
    trainingData = downloadData()

    docs, labels = read_training_data(trainingData)

    feature_fns = [token_features, token_pair_features, lexicon_features]
    results = eval_all_combinations(docs, labels, [True, False], feature_fns, [2, 5, 10])

    best_result = results[0]
    clf, vocab = fit_best_classifier(docs, labels, best_result)

    filePath = 'data/iphone.txt'
    testDocs = parse_tweets(filePath)

    testData, X = parse_test_data(best_result, vocab, testDocs)
    predictions = clf.predict(X)
    predictedProbabilities = clf.predict_proba(X)

    get_analyzed_result(predictions, predictedProbabilities, testData)

    endTime = datetime.datetime.now()
    print(endTime - startTime)


if __name__ == '__main__':
    main()
