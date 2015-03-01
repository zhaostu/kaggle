import csv
import random
from collections import defaultdict
from dt import id3

CLASS_ATTR = 'Survived'
IGNORE_ATTR = {'PassengerId', 'Name', 'Ticket'}

EVALUATE_ITERATIONS = 5

def read_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def clean(items):
    for item in items:
        # Get cabin class
        if len(item['Cabin']) > 0:
            item['Cabin'] = item['Cabin'][0]

        # Get fare class
        if len(item['Fare']) > 0:
            fare = float(item['Fare'])
            if fare == 0:
                item['Fare'] = 'free'
            elif fare <= 12:
                item['Fare'] = '0~12'
            elif fare <= 18:
                item['Fare'] = '12~18'
            elif fare <= 32:
                item['Fare'] = '18~32'
            elif fare <= 100:
                item['Fare'] = '32~100'
            elif fare <= 200:
                item['Fare'] = '100~200'
            elif fare <= 500:
                item['Fare'] = '200~500'
            else:
                item['Fare'] = '500~'

        # Get age class
        if len(item['Age']) > 0:
            age = float(item['Age'])
            if age <= 1:
                item['Age'] = 'infant'
            elif age <= 3:
                item['Age'] = 'toddler'
            elif age <= 12:
                item['Age'] = 'kid'
            elif age <= 18:
                item['Age'] = 'teenager'
            elif age <= 30:
                item['Age'] = 'young-adult'
            elif age <= 45:
                item['Age'] = 'mid-age'
            elif age <= 60:
                item['Age'] = 'old'
            else:
                item['Age'] = 'very-old'

        # Get the sibsp class
        if len(item['SibSp']) > 0:
            sibsp = int(item['SibSp'])
            if sibsp == 0:
                item['SibSp'] = 'none'
            elif sibsp == 1:
                item['SibSp'] = 'one'
            else:
                item['SibSp'] = 'more'

        # Get the parch class
        if len(item['Parch']) > 0:
            parch = int(item['Parch'])
            if parch == 0:
                item['Parch'] = 'none'
            elif parch == 1:
                item['Parch'] = 'one'
            elif parch == 2:
                item['Parch'] = 'two'
            else:
                item['Parch'] = 'more'

def evaluate(data, attrs, ratio):
    '''
    :param attrs: The attributes used for classifying.
    :param ratio: The ratio of training set to test set.
    '''
    matrix = defaultdict(int)
    num_training = int(ratio * len(data) / (1 + ratio))

    for i in range(EVALUATE_ITERATIONS):
        eval_data = list(data)
        random.shuffle(eval_data)
        training_set = eval_data[:num_training]
        testing_set = eval_data[num_training:]

        dtree = id3(training_set, attrs, CLASS_ATTR)
        for item in testing_set:
            predicted = dtree.make_decision(item)

            matrix[(item[CLASS_ATTR], predicted)] += 1

    print(matrix)

def main():
    training_set = read_csv('train.csv')
    clean(training_set)

    testing_set = read_csv('test.csv')
    clean(testing_set)

    attrs = frozenset(training_set[0].keys()) - {CLASS_ATTR} - IGNORE_ATTR

    # Evaluate the of the algorithm by using parts of the training data as
    # training
    evaluate(training_set, attrs, len(training_set) / len(testing_set))

    # Actual Training
    decision_tree = id3(training_set, attrs, CLASS_ATTR)

    with open('result.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['PassengerId', 'Survived'])
        writer.writeheader()

        for item in testing_set:
            survived = decision_tree.make_decision(item)
            writer.writerow({
                'PassengerId': item['PassengerId'],
                'Survived': survived})

if __name__ == '__main__':
    main()
