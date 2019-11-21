import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('mueller_report.csv')

dataset=dataset.dropna()
dataset = dataset.reset_index(drop=True)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
corpus=[]
for i in range(0,18747):
    review = re.sub('[^a-zA-Z_0-9]', ' ', dataset['text'][i])
    review=review.lower()
    review=review.split()
    ps=WordNetLemmatizer()
    review=[ps.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

test_list = [i for i in corpus if i] 


from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

# training model
model = Word2Vec(test_list,size=25, min_count=1, sg=1)
 
  
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
  
  
X2=[]
for sentence in test_list:
    X2.append(sent_vectorizer(sentence, model))   

X=np.array(X2,dtype=np.float32)
d=np.asarray(X2, dtype=np.float32)
X=X2.values

import matplotlib.pyplot as plt
wcss=[]
for i in range(1,4):
    kmeans=KMeans(n_clusters=i,init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,4),wcss)
plt.title('the elbow method')
plt.show()

clf=KMeans(n_clusters=2,max_iter=100,init='k-means++',n_init=1)
labels=clf.fit_predict(X)
print(labels)
for index,sentence in enumerate(sentence):
    print(str(labels[index]) + ":" + str(sentences))









































