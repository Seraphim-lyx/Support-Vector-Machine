
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string


train = pd.read_csv('train.csv')
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
train['toxic'].fillna(0, inplace =True)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()
comment = train['comment_text']
print(len(comment))

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
# test_term_doc = vec.transform(test[COMMENT])
x = trn_term_doc.A
y = train['toxic']
y = np.array(y).astype(int)
for i in range(len(y)):
	if y[i] == 0:
		y[i] = -1
print(y)
