from vncorenlp import VnCoreNLP
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing

annotator = VnCoreNLP("C:/Users/Thinh/KT-LAB/Week6/bai2/NLP Task/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
stopword = set(line.strip() for line in open('vietnamese-stopwords.txt',encoding="utf8"))
df = pd.read_csv('trainning.txt', header=None, encoding = "utf-8", delimiter ='\t')
df.columns =["Label", "content"]


def join_strings_best(strings):
    k = []
    for i in strings:
        l = ' '.join(str(x) for x in i)
        k.append(l)
    return ' '.join(k).replace(' .','.').replace(' ,',',')
def remove_stopwords(line):
    words = []
    for word in line.split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)


for lable,k in df.iterrows():
    lines = join_strings_best(annotator.tokenize(k['content']))
    line = remove_stopwords(lines)
    line = line.lower()
    new_df = pd.DataFrame({'content': [line]},index= [lable])
    df.update(new_df)

label = df["Label"]
content = df["content"]


data_trainning = pd.DataFrame({'Label': label,'Content': content})
data_trainning.to_csv('data_trainning.csv',index=False)

test = pd.read_csv('testing.txt', header=None, encoding = "utf-8", delimiter ='\t')
test.columns = ["content"]

for lable,k in test.iterrows():

    lines = join_strings_best(annotator.tokenize(k["content"]))
    line = remove_stopwords(lines)
    line = line.lower()
    new_df = pd.DataFrame({'content': [line]},index= [lable])
    test.update(new_df)

l = test['content']
data_testing = pd.DataFrame({'Content': l})
data_testing.to_csv('data_testing.csv',index=False)