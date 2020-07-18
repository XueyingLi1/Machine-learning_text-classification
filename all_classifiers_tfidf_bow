# coding: utf-8


# 需求评论二分类问题，使用了基于TF-IDF， Bow的向量化方法。
#NB，LR，DT，RF，SVM分类器。
#处理步骤是：导入数据集，分词，删除停用词（长度<=3的单词），提取tfidf，bow特征，训练分类器，将训练好的分类器应用于测试集。


import string
import pandas as pd
import numpy as np
import re
import gensim
from bs4 import BeautifulSoup
from nltk import word_tokenize
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import linear_model,naive_bayes,ensemble,svm,tree   #多分类器
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


#针对每一条评论说的
def review_to_wordlist(review):
    '''
    把评论转成词序列
    参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 小写化所有的词，并转成词list
    #review_text = review_text.lower()
    #print(type(review_text))
    review_final = ' '
    words1 = word_tokenize(review_text.lower())
    for w in words1:
        #删除长度<=3的单词
        if (len(w)>3):
            review_final = review_final + ' ' + w
            #print(review_final)
    words = review_final.split()
    #print(words)
    #words = review_text.split()
    # 返回words
    return words

#数据预处理，输出预处理后的训练集（及标签），测试集（及标签）
def data_preprocess():

  preprocess_begintime = datetime.datetime.now()
  # 载入数据集。数据集的格式为：id, catgory, review.其中，category为0或1
  train = pd.read_excel('train_data.xlsx', header=0, delimiter="\t")
  test = pd.read_excel('test_data.xlsx', header=0, delimiter="\t")


  # 训练集标签
  train_label = train['category']
  #测试集标签
  test_label = test['category'].tolist()
  # 存储预处理后的数据
  train_data = []
  #分词
  for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
  test_data = []
  for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

  # 预览数据 
  #print("train_data:",train_data[0], '\n')
  #print("test_data:",test_data[0])
  preprocess_endtime = datetime.datetime.now()
  
  #计算数据预处理时间，单位毫秒
  processtime = (preprocess_endtime - preprocess_begintime).seconds * 1000 + (preprocess_endtime - preprocess_begintime).microseconds / 1000 #微妙转换为毫秒
  print ("processtime:",processtime)
  return train_data,train_label,test_data,test_label,processtime



# ## 特征处理
# 直接丢给计算机这些词文本，计算机是无法计算的，因此我们需要把文本转换为向量，有几种常见的文本向量处理方法，比如：
# 1. 单词计数  
# 2. TF-IDF向量  
# 3. BoW向量  


# In[4]:

#输入：训练集，测试集数据。输出：训练集和测试集的tfidf
def tfidf_count(train_data,test_data):
  begintime = datetime.datetime.now()
  # 参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
  tfidf = TFIDF(min_df=2, # 最小支持度为2
           max_features=None, 
           strip_accents='unicode', #在预处理步骤中删除语气词。 
                                    #'ascii'是一种快速的方法，只适用于具有直接ASCII映射的字符。 
                                    #'unicode'是一种稍慢的方法，适用于任何字符。 None（默认）不起作用
           analyzer='word',         #特征值是一个单词 还是一个n-gram
           token_pattern=r'\w{1,}', #表示什么构成“token”的正则表达式，仅在分析器==“单词”时使用。 
                                    #默认正则表达式选择2个或更多字母数字字符的标记
                                    #（标点符号被完全忽略，并始终作为token分隔符处理）。
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,              #反文档频率
           smooth_idf=1,           #通过将文档频率添加一个平滑的idf权重，防止0频率
           sublinear_tf=1,         #用1 + log（tf）替换tf。
           stop_words=None) # 去掉英文停用词  如果是字符串，则将其传递给_check_stop_list，并返回相应的停止列表。
                                   #'english'是目前唯一支持的字符串值

  # 合并训练和测试集以便进行TFIDF向量化操作
  data_all = train_data + test_data
  len_train = len(train_data)

  tfidf.fit(data_all) #Learn vocabulary and idf from training set. 

  data_all = tfidf.transform(data_all)  #Transform documents to document-term matrix


  #print(data_all)

  # 恢复成训练集和测试集部分
  train_tfidf = data_all[:len_train] 
  test_tfidf = data_all[len_train:]
  print('TF-IDF over.')

  endtime = datetime.datetime.now()
  tfidftime = (endtime - begintime).seconds * 1000 + (endtime - begintime).microseconds / 1000 #微妙转换为毫秒

  return train_tfidf, test_tfidf, tfidftime

def bow(train_data,test_data):
  begintime = datetime.datetime.now()
  bow = CountVectorizer()
  data_all = train_data + test_data
  len_train = len(train_data)
  data_all = bow.fit_transform(data_all)

  train_bow = data_all[:len_train]
  test_bow = data_all[len_train:]
  print ('Bow  over.')

  endtime = datetime.datetime.now()
  bowtime = (endtime - begintime).seconds * 1000 + (endtime - begintime).microseconds / 1000 #微妙转换为毫秒

  return train_bow,test_bow,bowtime

#使用多种分类器，输出每种分类器的性能得分
def train_classifier():

  #预处理
  train_data,train_label,test_data,test_label,processtime = data_preprocess()
  #计算tfidf值
  train_tfidf,test_tfidf, tfidftime = tfidf_count(train_data,test_data)
  # 计算Bow值
  train_bow,test_bow,bowtime = bow(train_data,test_data)

  features = [('tfidf',train_tfidf,test_tfidf,tfidftime),('bow',train_bow,test_bow,bowtime)]
    

  #此处使用GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。
  LR_model = GridSearchCV(cv=10, estimator=linear_model.LogisticRegression(C=1.0, class_weight=None, dual=False,
             fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=0, tol=0.0001),
       iid=True, n_jobs=1,
        param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
        scoring='roc_auc', verbose=0)

  #定义多个分类器
  classifiers = []
  classifiers.append(('DT',tree.DecisionTreeClassifier()))
  classifiers.append(('LR',LR_model))
  classifiers.append(('NB',naive_bayes.MultinomialNB()))
  classifiers.append(('RF',ensemble.RandomForestClassifier()))
  classifiers.append(('SVM',svm.SVC(kernel='linear')))

  #存储评估结果
  results = []

  #训练分类器,基于tfidf
  for name,classifier in classifiers:
    for name_feature,train_count,test_count,time in features:
      train_begintime = datetime.datetime.now()
      #print (name_feature,train_count,test_count)
      classifier.fit(train_count, train_label)
      train_endtime = datetime.datetime.now()
      #10折交叉验证得分
      #print("10-cross score:", np.mean(cross_val_score(classifier, train_tfidf, train_label, cv=10, scoring='accuracy')))
      test_begintiem =  datetime.datetime.now()
      test_predicted = classifier.predict(test_count)

      #计算precision,recall,f1-score
      print(name+' '+name_feature)
      print(accuracy_score(test_label, test_predicted),
        precision_score(test_label, test_predicted),
        recall_score(test_label, test_predicted),
       f1_score(test_label, test_predicted, average='binary'))
      
      test_endtime = datetime.datetime.now()

      
      classfiertime = (train_endtime - train_begintime).seconds * 1000 + (train_endtime - train_begintime).microseconds / 1000 #微妙转换为毫秒

      #使用一个分类器处理一次的时间为：预处理时间+特征提取时间+训练预测时间
      totaltime = processtime+time+classfiertime
      print ("totaltime:",totaltime)
      # print (name,'\n')
      # 建模耗时
      print ("trainTime:",((train_endtime - train_begintime).seconds * 1000 + (train_endtime - train_begintime).microseconds / 1000),(train_endtime - train_begintime).seconds)
      # 模型预测耗时
      print ("testTime:",((test_endtime - test_begintiem).seconds * 1000 + (test_endtime - test_begintiem).microseconds / 1000),(test_endtime - test_begintiem).seconds)

      print ('\n')


if __name__ == '__main__':
  train_classifier()
