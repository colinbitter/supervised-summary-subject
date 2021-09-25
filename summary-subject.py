import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import zero_one_loss
from sklearn.svm import LinearSVC
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from pathlib import Path
import glob
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path1 = str(Path.home() / "Downloads")
allFiles = glob.glob(path1 + "/*.txt")

print("===READ IN===")
print(datetime.now())
# read in data - output from marcedit is 001, 245, 520, 650_0
# in field delimiter use @ - important for 650_0
for file_ in allFiles:
    df = pd.read_csv(file_, sep="\t", header=0, encoding='ISO-8859-1')

print("===PREPROCESS===")
print(datetime.now())
# preprocessing MARC oddities
df.rename(columns={df.columns[0]: 'LocalID', df.columns[1]: 'Title', df.columns[2]: 'SummaryNative',
                   df.columns[3]: 'Subject'}, inplace=True)
df['Summary'] = df['Title'] + ' ' + df['SummaryNative']
df['Summary'] = df['Summary'].replace({r'\\\\': ''}, regex=True)
df['Summary'] = df['Summary'].replace({r'\d\\': ''}, regex=True)
df['Summary'] = df['Summary'].replace({r'\d\\$\w': ''}, regex=True)
df['Summary'] = df['Summary'].replace({r'\$\w': ' '}, regex=True)
df['Summary'] = df['Summary'].replace({r'880\-0\d': ''}, regex=True)
df['Subject'] = df['Subject'].replace({r'\.': ''}, regex=True)
df['Subject'] = df['Subject'].replace({r'\\0\$\w': ''}, regex=True)
df['Subject'] = df['Subject'].replace({r'\$\w': ' '}, regex=True)
df['Subject'] = df['Subject'].replace({r'880\-0\d': ''}, regex=True)

# drop
df = df.dropna()

# # split subjects on @
df['Subject'] = df['Subject'].str.split("@")

# nltk
stopword_list = nltk.corpus.stopwords.words('english')

print("===NORMALIZE===")
print(datetime.now())


def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    doc = ' '.join(filtered_tokens)
    return doc


df['Summary'] = [normalize_document(c) for c in df['Summary']]

# drop
df = df.dropna()

dfMCC = df

print("===MULTICLASS CLASSIFICATION===")
print(datetime.now())
# select first subject from each record for multiclass classification
dfMCC['SubjectSingle'] = dfMCC['Subject'].str[0]
# eliminate labels with less than 100 occurrences
dfMCC = dfMCC[dfMCC.groupby('SubjectSingle').LocalID.transform(len) > 100]
dfMCC = dfMCC.dropna()

# MCC checkpoint
dfMCC.to_csv('MCCcheckpoint.csv', index=False)

print("===TRAIN TEST SPLIT===")
print(datetime.now())
# target variable
y = dfMCC['SubjectSingle']
# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(dfMCC['Summary'], y, test_size=0.2, random_state=9)  # test size?

print("===TFIDF===")
print(datetime.now())
tfv = TfidfVectorizer()  # can alter min max df
xtrain1 = tfv.fit_transform(xtrain)
xval1 = tfv.transform(xval)

print("===SUPPOR VECTOR CLASSIFICATION===")
print(datetime.now())
svm = OneVsRestClassifier(LinearSVC(random_state=9))
svm.fit(xtrain1, ytrain)
y_pred = svm.predict(xval1)

print("===OneVsRest SVC===")
print(datetime.now())
svm_tfidf_test_score = svm.score(xtrain1, ytrain)

print('Test Accuracy:', svm_tfidf_test_score)
print("Accuracy = ", accuracy_score(yval, y_pred))
print("Classification report = ", classification_report(yval, y_pred, zero_division=0))
print("F1 micro = ", f1_score(yval, y_pred, average="micro"))
print("F1 macro = ", f1_score(yval, y_pred, average="macro"))
print("F1 weighted = ", f1_score(yval, y_pred, average="weighted"))
print("F-beta micro = ", fbeta_score(yval, y_pred, average="micro", beta=0.5))
print("F-beta macro = ", fbeta_score(yval, y_pred, average="macro", beta=0.5))
print("F-beta weighted = ", fbeta_score(yval, y_pred, average="weighted", beta=0.5))
print("Haming loss = ", hamming_loss(yval, y_pred))
print("Jaccard micro = ", jaccard_score(yval, y_pred, average="micro"))
print("Jaccard macro = ", jaccard_score(yval, y_pred, average="macro"))
print("Jaccard weighted = ", jaccard_score(yval, y_pred, average="weighted"))
print("Precision micro = ", precision_score(yval, y_pred, average="micro"))
print("Precision macro = ", precision_score(yval, y_pred, average="macro"))
print("Precision weighted = ", precision_score(yval, y_pred, average="weighted"))
print("Recall micro = ", recall_score(yval, y_pred, average="micro"))
print("Recall macro = ", recall_score(yval, y_pred, average="macro"))
print("Recall weighted = ", recall_score(yval, y_pred, average="weighted"))
print("Zero-one loss = ", zero_one_loss(yval, y_pred))

print("===END===")
print(datetime.now())
