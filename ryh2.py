import pandas as pd
import  numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

df = pd.read_csv(r"C:\Users\mfarr\Desktop\PycharmProjects\movie_data.csv", encoding= 'latin1')
# print(df.head(10))
count = CountVectorizer()

docs = np.array(['the sun is shining',
'the weather is sweet',
'the sun is shining, the weather is sweet, and one and one is two'])

bag = count.fit_transform(docs)

# print(count.vocabulary_)

# print(bag.toarray())

# {'the': 6, 'sun': 4, 'is': 1, 'shining': 3, 'weather': 8, 'sweet': 5, 'and': 0, 'one': 2, 'two': 7}
# [[0 1 0 1 1 0 1 0 0]
#  [0 1 0 0 0 1 1 0 1]
#  [2 3 2 1 1 1 2 1 1]]

# print(df['review'][0])
np.set_printoptions(precision=2)

tfidf = TfidfTransformer(use_idf= True, norm='l2',  smooth_idf=True)
# print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#'is' is now less weight, thus contains less useful, core idea of TFIDF values

# print(df.loc[0, 'review'][-50:])

#trying to get rid of the <.br and such

def preprocessor(text):
    text = re.sub('<[^>]*>','',text)
    emojis = re.findall('(?::|;|=)(?:-)?(?:-\)|\(|D|P)',text)
    text = re.sub('[\W]+',' ',text.lower()) +\
        ''.join(emojis).replace('-','')
    return  text

print (preprocessor(df.loc[0, 'review'][-50:]))
#everything stripped, now its just 'is seven title brazil not available' as final 50 char


#print (preprocessor("</a>This :) is a :( test :)! "))
#just an example


df['review'] = df['review'].apply(preprocessor)

porter = PorterStemmer()

def tokenizer (text):
    return text.split()

def tokenizer_stemmer(text):
    return [porter.stem(word) for word in text.split()]


print(tokenizer('runners like running and thus they run'))

print(tokenizer_stemmer('runners like running and thus they run'))

# nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_stemmer('a runner likes running and runs a lot')[-10:] if w not in stop])


#sheds words and tokenizes them

tfidf  = TfidfVectorizer(strip_accents= None,
                         lowercase= False,
                         preprocessor=None,
                         tokenizer=tokenizer_stemmer,
                         use_idf=True,
                         norm = 'l2',
                         smooth_idf = True)

y = df.sentiment.values
x = tfidf.fit_transform(df.review)

#split data into training and test sets, use helper functon

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state= 1, test_size= 0.5,
                                                shuffle=False)

#logistic regression

import pickle
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv  = 5,
                           scoring= 'accuracy',
                           random_state=0,
                           n_jobs=-1,
                           verbose=3,
                           max_iter=300).fit(xtrain,ytrain)

saved_model = open('saved_model.sav', 'wb')
pickle.dump(clf, saved_model)
saved_model.close()

#pickle for saving
#300 for safety


#load saved model

filename = 'saved_model.sav'
saved_clf = pickle.load(open(filename, 'rb'))

saved_clf.score(xtest, ytest)


#https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression




