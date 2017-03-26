import re
import time
from collections import defaultdict
from itertools import combinations
import nltk

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

appleInc_words = set()
appleFruit_words = set()
docs = list()
labels = list()

def readData():
    a = open('appleInc.txt', encoding="utf8")
    for line in a:
        docs.append((1, line))
        tokenList = tokenize(line, True)
        appleInc_words.update(tokenList)
    f = open('fruit.txt', encoding="utf8")
    for line in f:
        docs.append((0, line))
        tokenList = tokenize(line, True)
        appleFruit_words.update(tokenList)
    return np.array([d[1] for d in docs]), np.array([d[0] for d in docs])


def tokenize(doc, keep_internal_punct=True):
    tokenList = []
    if keep_internal_punct == False:
        return np.array(re.sub('\W', ' ', doc).lower().strip().split())
    if keep_internal_punct == True:
        splitResult = doc.split(' ')
        for word in splitResult:
            word = re.sub("[\W]+$", '', word, flags=re.UNICODE)
            word = re.sub("^[\W]+", '', word, flags=re.UNICODE)
            if len(word) > 0:
                tokenList.append(word.lower().strip())
        return np.array(tokenList)


def token_pair_features(tokens, feats, k=3):
    for i in range(0, len(tokens)):
        if i + k <= len(tokens):
            subSet = tokens[i:i + k]
            for j in range(0, len(subSet) - 1):
                for l in range(j + 1, len(subSet)):
                    if ('token_pair=' + subSet[j] + '__' + subSet[l]) in feats:
                        feats['token_pair=' + subSet[j] + '__' + subSet[l]] = (
                            feats['token_pair=' + subSet[j] + '__' + subSet[l]] + 1)
                    else:
                        feats['token_pair=' + subSet[j] + '__' + subSet[l]] = 1


def lexicon_features(tokens, feats):
    feats['appleInc_words'] = 0
    feats['appleFruit_words'] = 0
    for token in tokens:
        if token in [x for x in appleInc_words]:
            feats['appleInc_words'] = (feats['appleInc_words'] + 1)
        if token in [x for x in appleFruit_words]:
            feats['appleFruit_words'] = (feats['appleFruit_words'] + 1)


def featurize(tokens, feature_fns):
    featureList = []
    for funcDef in feature_fns:
        feats = dict()
        funcDef(tokens, feats)
        for items in feats.items():
            featureList.append(items)
    return sorted(featureList, key=lambda tup: tup[0])


def vectorize(tokens_list, feature_fns, vocab=None):
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
        finalVocabList = dict((k, v) for k, v in vocabList.items())
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


def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation, :]
    Y2 = Y[permutation]
    return X2, Y2


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(len(labels), k)
    accuracies = []
    for train_idx, test_idx in cv:
        if len(set(labels[train_idx])) == 1:
            X2, Y2 = randomize(X, labels)
            clf.fit(X2[train_idx], Y2[train_idx])
        else:
            clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns):
    listOfDict = []
    clf = LogisticRegression()
    for punctuation in punct_vals:
        for L in range(0, len(feature_fns) + 1):
            for subset in combinations(feature_fns, L):
                if subset:
                    X, vocabs = vectorize([tokenize(d, punctuation) for d in docs], subset, vocab=None)
                    accurracy = cross_validation_accuracy(clf, X, np.array(labels), 2)
                    listOfDict.append(
                        {'features': subset, 'punct': punctuation, 'accuracy': accurracy})
    listOfDict.sort(key=lambda x: (-x['accuracy']))
    return listOfDict


def fit_best_classifier(docs, labels, best_result):
    punct = best_result['punct']
    features = best_result['features']
    X, vocab = vectorize([tokenize(d, punct) for d in docs], features, vocab=None)
    clf = LogisticRegression()
    clf.fit(X, labels)
    return (clf, vocab)


def parse_test_data(best_result, vocab):
    testDocs = list()
    a = open('input.txt', encoding="utf8")
    for line in a:
        testDocs.append(line)
    punct = best_result['punct']
    features = best_result['features']
    X, vocab = vectorize([tokenize(d, punct) for d in testDocs], features, vocab)
    return testDocs, X


def main():
    start_time = time.time()
    docs, labels = readData()
    feature_fns = [token_pair_features, lexicon_features]
    # Evaluate accuracy of many combinations of tokenization/featurization.
    # results = eval_all_combinations(docs, labels,
    #                                [True, False],
    #                                feature_fns)
    # best_result = results[0]
    """I have evaluated the accuracy using above function and found the best result for below settings
    Thus I have directly used these settings to train my model and used for predictions.
    """
    best_result = dict()
    best_result['punct'] = True
    best_result['features'] = [lexicon_features]
    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, best_result)

    # Parse test data
    test_docs, X_test = parse_test_data(best_result, vocab)
    print("--- %s seconds ---" % (time.time() - start_time))

    for i in clf.predict(X_test):
        if i == 1:
            print("computer-company")
        else:
            print("fruit")


if __name__ == '__main__':
    main()
