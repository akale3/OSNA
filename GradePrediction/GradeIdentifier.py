import json
import time
from random import sample
import threading

import numpy as np
import pandas as pd
from sklearn import linear_model

ENPHCHCS = ["Chemistry", "ComputerScience", "English", "Physics"]
ENPHCHCS_list = list()
ENPHCHPE = ["Chemistry", "English", "PhysicalEducation", "Physics"]
ENPHCHPE_list = list()
ENPHCHEO = ["Chemistry", "Economics", "English", "Physics"]
ENPHCHEO_list = list()
ENPHCHBI = ["Biology", "Chemistry", "English", "Physics"]
ENPHCHBI_list = list()
ENECACBS = ["Accountancy", "BusinessStudies", "Economics", "English"]
ENECACBS_list = list()
other = list()


def make_predictions(lineDf, matchedSubRecords, subjectsList):
    for outerIndex, testRow in lineDf.iterrows():
        vectorA = testRow[subjectsList]
        cosineSum, meanRating, cosineRatingSum = 0, 0, 0
        noOfStudents = 1
        if len(matchedSubRecords) >= 1000:
            maxRecords = 1000
        else:
            maxRecords = len(matchedSubRecords)
        rindex = np.array(sample(range(len(matchedSubRecords)), maxRecords))
        dfr = matchedSubRecords.ix[rindex]
        for index, trainingRow in dfr.iterrows():
            vectorB = trainingRow[subjectsList]
            if abs(np.subtract(vectorA, vectorB).values).sum() <= 1:
                cosineRatingSum += trainingRow.Mathematics
                noOfStudents += 1
        marks = int(round(cosineRatingSum / noOfStudents))
        if marks == 0:
            print(1)
        else:
            print(marks)


def readData():
    a = open('training.json', encoding="utf8")
    for line in a:
        if all(x in line for x in ENPHCHCS):
            ENPHCHCS_list.append(line)
        elif all(x in line for x in ENPHCHPE):
            ENPHCHPE_list.append(line)
        elif all(x in line for x in ENPHCHEO):
            ENPHCHEO_list.append(line)
        elif all(x in line for x in ENPHCHBI):
            ENPHCHBI_list.append(line)
        elif all(x in line for x in ENECACBS):
            ENECACBS_list.append(line)
        else:
            other.append(line)
    ENPHCHCS_df = pd.DataFrame.from_records(map(json.loads, ENPHCHCS_list))
    ENPHCHPE_df = pd.DataFrame.from_records(map(json.loads, ENPHCHPE_list))
    ENPHCHEO_df = pd.DataFrame.from_records(map(json.loads, ENPHCHEO_list))
    ENPHCHBI_df = pd.DataFrame.from_records(map(json.loads, ENPHCHBI_list))
    ENECACBS_df = pd.DataFrame.from_records(map(json.loads, ENECACBS_list))
    other_df = pd.DataFrame.from_records(map(json.loads, other))
    return ENPHCHCS_df, ENPHCHPE_df, ENPHCHEO_df, ENPHCHBI_df, ENECACBS_df, other_df


def worker(startIndex, endIndex, listList, lineList, ENPHCHCS_clf, ENPHCHPE_clf, ENPHCHEO_clf, ENPHCHBI_clf,
           ENECACBS_clf):
    for x in range(startIndex, endIndex):
        rowList = []
        line = lineList[x]
        rowList.append(line)
        if all(x in line for x in ENPHCHCS):
            listList.append(ENPHCHCS_clf.predict(pd.DataFrame.from_records(map(json.loads, rowList))[ENPHCHCS])[0])
        elif all(x in line for x in ENPHCHPE):
            listList.append(ENPHCHPE_clf.predict(pd.DataFrame.from_records(map(json.loads, rowList))[ENPHCHPE])[0])
        elif all(x in line for x in ENPHCHEO):
            listList.append(ENPHCHEO_clf.predict(pd.DataFrame.from_records(map(json.loads, rowList))[ENPHCHEO])[0])
        elif all(x in line for x in ENPHCHBI):
            listList.append(ENPHCHBI_clf.predict(pd.DataFrame.from_records(map(json.loads, rowList))[ENPHCHBI])[0])
        elif all(x in line for x in ENECACBS):
            listList.append(ENECACBS_clf.predict(pd.DataFrame.from_records(map(json.loads, rowList))[ENECACBS])[0])
        else:
            # print(other_clf.predict(pd.DataFrame.from_records(map(json.loads, lineList))[other]))
            listList.append(4)


def readTestData(ENPHCHCS_df, ENPHCHPE_df, ENPHCHEO_df, ENPHCHBI_df, ENECACBS_df, other_df):
    ENPHCHCS_clf = linear_model.LinearRegression()
    ENPHCHCS_clf.fit(ENPHCHCS_df[ENPHCHCS][:500], ENPHCHCS_df["Mathematics"][:500])

    ENPHCHPE_clf = linear_model.LinearRegression()
    ENPHCHPE_clf.fit(ENPHCHPE_df[ENPHCHPE][:500], ENPHCHPE_df["Mathematics"][:500])

    ENPHCHEO_clf = linear_model.LinearRegression()
    ENPHCHEO_clf.fit(ENPHCHEO_df[ENPHCHEO][:500], ENPHCHEO_df["Mathematics"][:500])

    ENPHCHBI_clf = linear_model.LinearRegression()
    ENPHCHBI_clf.fit(ENPHCHBI_df[ENPHCHBI][:500], ENPHCHBI_df["Mathematics"][:500])

    ENECACBS_clf = linear_model.LinearRegression()
    ENECACBS_clf.fit(ENECACBS_df[ENECACBS][:500], ENECACBS_df["Mathematics"][:500])

    # other_clf = linear_model.LinearRegression() ()
    # other_clf.fit(other_df[other], other_df["Mathematics"])

    lineList = []
    a = open('sample-test.in1.json', encoding="utf8")
    for line in a:
        lineList.append(line)
    listList = [[], [], [], [], [], [], [], [],[],[]]
    startValue = len(lineList) / 10
    for i in range(10):
        t = threading.Thread(
            target=worker(i * int(startValue), ((i + 1) * int(startValue)), listList[i], lineList, ENPHCHCS_clf,
                          ENPHCHPE_clf,
                          ENPHCHEO_clf, ENPHCHBI_clf, ENECACBS_clf))
        t.start()
    for x in listList:
        for y in x:
            print(int(y))


def main():
    start_time = time.time()
    ENPHCHCS_df, ENPHCHPE_df, ENPHCHEO_df, ENPHCHBI_df, ENECACBS_df, other_df = readData()
    print("--- %s seconds ---" % (time.time() - start_time))
    # studentRecords, vocab = featurize(studentRecords)
    readTestData(ENPHCHCS_df, ENPHCHPE_df, ENPHCHEO_df, ENPHCHBI_df, ENECACBS_df, other_df)
    # predictedMarks = make_predictions(studentRecords, studentTest)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
