import math
from collections import defaultdict

def entropy(data, attr):
    ''' Find out the entropy over a certain attribute of the given data.

    :param data: A list of items, with each item represented as dict.
    :param attr: The attribute use for calculating entropy. item[attr] must
    be a hashable object.
    '''
    bucket = defaultdict(int)
    n = len(data)
    for item in data:
        a = item[attr]
        bucket[a] += 1

    h = 0

    for c in bucket.values():
        p = c / n
        h += -p * math.log2(p)

    return h
