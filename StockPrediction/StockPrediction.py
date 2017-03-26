import csv


def getPreviousValue(rows, i):
    missingEntrycount = 0
    while (i >= 2 and 'Missing_' in str(rows[i][1])):
        i -= 1
        missingEntrycount += 1
    return rows[i][1], missingEntrycount


def getNextValue(rows, i):
    j = i
    missingEntrycount = 0
    while ('Missing_' in str(rows[i][1]) and i < len(rows) - 1):
        i += 1
        missingEntrycount += 1
    if 'Missing_' in rows[i][1]:
        return rows[j - 1][1], missingEntrycount
    else:
        return rows[i][1], missingEntrycount


def get_predicted_value(previousValue, nextValue, missing_entries_count):
    value_diff = previousValue - nextValue
    if value_diff > 0:
        return previousValue - (abs(value_diff) / (missing_entries_count + 1))
    else:
        return previousValue + (abs(value_diff) / (missing_entries_count + 1))


def main():
    print("Predicting Stock values")
    with open('input001.txt', 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        rows = list()
        for x in reader:
            rows.append(x)

        missingValues = list()
        previousValue = 0
        nextValue = 0
        for i in range(1, len(rows)):
            if 'Missing_' in rows[i][1]:
                if i != 1:
                    previousValue, miss_prev_count = getPreviousValue(rows, i)
                else:
                    previousValue, miss_prev_count = getNextValue(rows, i)

                if i != len(rows) - 1:
                    nextValue, miss_next_count = getNextValue(rows, i)
                else:
                    nextValue, miss_next_count = getPreviousValue(rows, i)

                rows[i][1] = get_predicted_value(float(previousValue), float(nextValue),
                                                 (miss_next_count + miss_prev_count))
                missingValues.append(rows[i][1])

        for missingValue in missingValues:
            print(missingValue)


if __name__ == '__main__':
    main()
