import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()

docs = np.array(['the sun is shining',
'the weather is sweet',
'the sun is shining, the weather is sweet, and one and one is two'])

bag = count.fit_transform(docs)

print(count.vocabulary_)

print(bag.toarray())

# {'the': 6, 'sun': 4, 'is': 1, 'shining': 3, 'weather': 8, 'sweet': 5, 'and': 0, 'one': 2, 'two': 7}
# [[0 1 0 1 1 0 1 0 0]
#  [0 1 0 0 0 1 1 0 1]
#  [2 3 2 1 1 1 2 1 1]]

