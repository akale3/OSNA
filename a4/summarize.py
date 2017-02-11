"""
sumarize.py
"""
import datetime
import os


def readUserFile(filePath):
    resultList = list()
    f = open(filePath, 'r')
    for line in f:
        if len(line) > 0:
            resultList.append(line)
    return resultList


def write_content_to_summary_file(filePath):
    resultList = readUserFile(filePath)
    with open('summary.txt', 'a') as summaryFile:
        for data in resultList:
            summaryFile.write(data)


def main():
    startTime = datetime.datetime.now()

    try:
        os.remove('summary.txt')
    except Exception as e:
        print(e)

    filePath = 'cluster.txt'
    write_content_to_summary_file(filePath)

    filePath = 'classify.txt'
    write_content_to_summary_file(filePath)

    endTime = datetime.datetime.now()
    print(endTime - startTime)


if __name__ == '__main__':
    main()
