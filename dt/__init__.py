from collections import defaultdict

from .entropy import entropy

MIN_ENTROPY = 0.70 # Entropy for something happens p=0.05

class DecisionTree:
    def __init__(self, attr, most_likely):
        self.attr = attr
        self.most_likely = most_likely
        self.children = dict()

    def add_child(self, cond, dt):
        self.children[cond] = dt

    def make_decision(self, item):
        a = item[self.attr]

        if a not in self.children:
            # Has not seen this branch yet, use the most likely result.
            return self.most_likely

        if isinstance(self.children[a], DecisionTree):
            return self.children[a].make_decision(item)
        else:
            return self.children[a]

def most_likely(data, class_attr):
    bucket = defaultdict(int)
    for item in data:
        bucket[item[class_attr]] += 1

    max_n = 0
    max_class = None
    
    for c, n in bucket.items():
        if n > max_n:
            max_n = n
            max_class = c

    return max_class

def id3(data, attrs, class_attr):
    ''' Build a decision tree using ID3 algorithm.

    :param data: A list of items, with each item represented as dict.
    :param attrs: A set of attributes that can be used to make decision.
        This should not include class_attr.
    :param class_attr: The attribute used as classifier.
    '''
    n = len(data)
    h = entropy(data, class_attr)
    best_attr = None
    best_bucket = None
    best_ig = None

    most_likely_result = most_likely(data, class_attr)

    if h < MIN_ENTROPY:
        return most_likely_result

    # Find out which attributes gives the best information gain.
    for attr in attrs:
        bucket = defaultdict(list)
        data_with_value = [d for d in data if d[attr] != '']
        n = len(data_with_value)

        # Group data with value by its corresponding attribute.
        for item in data_with_value:
            bucket[item[attr]].append(item)

        h_sub = 0
        for a, items in bucket.items():
            if a == '':
                continue
            h_sub += (len(items) / n) * entropy(items, class_attr)

        if best_ig is None or h - h_sub > best_ig:
            best_attr = attr
            best_bucket = bucket
            best_ig = h - h_sub

    if best_attr is not None:
        dt = DecisionTree(best_attr, most_likely_result)
        print('Decision Tree based on %s.' % best_attr)
        for a, items in best_bucket.items():
            print('  Finding child for %s=%s with %d items.' % (best_attr, a, len(items)))
            dt.add_child(a, id3(items, attrs-{best_attr}, class_attr))
    else:
        return most_likely_result

    return dt
