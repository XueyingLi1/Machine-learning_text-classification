#import train
#代码框架：参数配置，训练数据生成，模型结构，训练模型
import os
import csv
import time
import datetime
import random
import json
import re
from bs4 import BeautifulSoup

from collections import Counter
from math import sqrt
from typing import Dict, Any, Union

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


#设置系统打印信息：为2时，打印Error和Fatal信息。屏蔽info和warning信息。
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


beginTime = datetime.datetime.now()

#配置参数
class TrainingConfig(object):
    print ("TrainingConfig():",1)
    epoches = 10
    evaluateEvery = 15       #训练集样本数1022 // batchsize = 15。每完成一次训练，进行评估。
    checkpointEvery = 15     #每15轮保存模型
    learningRate = 0.001

class ModelConfig(object):
    print ("ModelConfig():",2)
    embeddingSize = 200   #词向量长度
    numFilters = 128 #卷积核数量

    filterSizes = [3,4,5] #卷积核尺寸
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    print ("config():",3)
    sequenceLength = 200  #词序列长度
    batchSize = 64

    dataSource = "dataSet0320.xlsx"
    stopWordSource = "StopWords.txt"
    numClasses = 1 #二分类设置为1，多分类设置为类别的数目
    rate = 8/9 #训练集的比例
    training = TrainingConfig()
    model = ModelConfig()

#实例化配置参数对象
config = Config()


#数据预处理的类，生成训练集和测试集
#_data，单下划线的变量表示的是protected类型的变量。只允许类本身和子类访问。
class Dataset(object):
    #类的构造函数/初始化方法，在创建类的实例时调用该方法。
    #self代表类的实例（必须有的参数，代表当前对象的地址）
    #self.xxx，实例变量xxx
    #config参数：在实例化Dataset类时，必须提供config
    def __init__(self,config):
        #print ("init:",config)
        self.config = config
        self._dataSource = config.dataSource         #数据源
        self._stopWordSource = config.stopWordSource #停用词

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize #词向量维度
        self._batchSize = config.batchSize              
        self._rate = config.rate                   #训练集比例

        self._stopWordDict = {}                   #在_readStopWord()函数中被赋值

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None              #在_genVocabulary()函数中被赋值。存储：删除停用词以及低频词后的，其他单词的词向量

        self.labelList = []                    #在_genVocabulary()函数中被赋值。存储：[0,1]

    def _readData(self, filePath):
        """
        从excel文件中读取数据集
        """
        print ("_readData():",6)
        df = pd.read_excel(filePath)

        if self.config.numClasses == 1: #二分类
            labels = df["category"].tolist() #为什么要变成list？
            #print ("labels:",labels)
        elif self.config.numClasses > 1: #多分类
            labels = df["rate"].tolist()

        review = df["review"].tolist()
        #print ("review:",review)
        #分词
        reviews = [line.strip().split() for line in review]
        #print ("reviews:",reviews)
        #返回所有词以及句子标签
        return reviews, labels

    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        print ("_labelToIndex():step",8,'\n')
        labelIds = [label2idx[label] for label in labels]
        """
        for label in labels:
            label2idx[label]
        """

        #print("labelIds:",labelIds,'\n')
        return labelIds

    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        print ("_wordToIndex():step",9,'\n')
        #dict.get(key,default=None):返回dict中，key对应的vaule，如果key不存在于dict中，返回默认值。
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        #print ("reviewIds:",reviewIds,'\n')
        """
        for review in reviews:
            for item in review:
                word2idx.get(item, word2idx["UNK"])
        """
        return reviewIds

    def _genTrainEvalData(self, x, y, word2idx, rate):
        """
        生成训练集和验证集
        """
        #x:以单词索引构成的dict，y:以标签索引构成的数组，word2idx:词汇-索引表，rate:数据分割
        print ("_genTrainEvalData():step",10,'\n')
        #存放由单词索引构成的句子。每个句子长度为200
        reviews = []
        for review in x:
            #句子长度是否大于200
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                #句子长度不足200时，用0填充。
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))
        #训练集大小：原始数据集大小*rate
        trainIndex = int(len(x) * rate)
        #a[n:],截取从n后的所有元素，包含n位置。a[:n],截取从0开始到n-1的所有元素，不包含n位置。
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        #print("trainReviews:",trainReviews)

        trainLabels = np.array(y[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")
        # trainReviews:以单词索引构成的句子。
        # trainlabels: 以标签索引构成。
        return trainReviews, trainLabels, evalReviews, evalLabels

    #生成词向量和词汇-索引映射字典，可以用全数据集
    def _genVocabulary(self, reviews, labels):
        print ("_genVocabulary():",7)

        #数据集中的所有词汇
        #print ("reviews:",len(reviews))
        allWords = [word for review in reviews for word in review]
        print ("allWords:",len(allWords))
        """
        for review in reviews:
            for word in review:
                word
        """

        # 去掉停用词 并且去掉长度小于等于3的词
        subWords = [word for word in allWords if((word not in self.stopWordDict) and len(word) > 3)]
        print ("subWords:",len(subWords))
        print ("remove stop words...over")
        """
        for word in allWords:
            if ((word not in self.stopWordDict) and len(word) > 3):
                word
        """

        wordCount = Counter(subWords)  # 统计所有单词的词频
        #sorted方法返回一个新的列表。key：指定用来进行比较的元素。reverse:排序规则，True降序，False升序。
        #lambda x:x[1]，表示对wordCount.items中的第二维数据（即value）的值进行排序。
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        print ("sorted words...over")
        #print ("sortWordCount:",sortWordCount)  #形如：[('have', 119), ('this', 117), ('video', 108), ...]

        # 去除低频词，删除频率低于5的单词
        words = [item[0] for item in sortWordCount if item[1] >= 5]
        print ("words:",len(words))
        print ("remove words appear less than 5...over")

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
        print ("get vocab_wordVec and wordEmbedding...over")

        #单词-索引
        word2idx = dict(zip(vocab, list(range(len(vocab)))))
        #print ("word2idx:",word2idx)
        #print(word2idx) right

        #标签-索引
        uniqueLabel = list(set(labels))
        #print(uniqueLabel) right
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        print("label2idx:",type(label2idx),label2idx)  #{'P': 0, 'N': 1}
        self.labelList = list(range(len(uniqueLabel)))
        #print (self.labelList)  #[0, 1]


        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("word2idx.json", "w", encoding="utf-8") as f:
            json.dump(word2idx, f) #将转换完格式后的字符串写入文件
        #print(word2idx)

        with open("label2idx.json", "w", encoding="utf-8") as f:
            json.dump(label2idx, f)
        #word2idx:{'PAD': 0, 'UNK': 1, 'have': 2, 'this': 3, 'video': 4, 'when': 5, 'with': 6,...}
        #label2idx:{'P': 0, 'N': 1}
        print ('\n')
        return word2idx, label2idx

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2vec.bin", binary=True)
        #print("wordvec_type:",type(wordVec)) #type: gensim.models.keyedvectors.Word2VecKeyedVectors
        #可查看单个词的词向量
        #print ("wordvec_model:",wordVec['video'])
        #print ("wordvec_model:",wordVec.wv['video'])
        #存储单词
        vocab = []
        #存储词向量
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("PAD") #append方法：在list末尾添加元素。无返回值，但是会修改原来的列表
        vocab.append("UNK") #有些词可能不在word2vec词向量中，这些词使用UNK表示
        wordEmbedding.append(np.zeros(self._embeddingSize))  #np.zeros(5),用0初始化数组.[0,0,0,0,0],此处为np.zeros(200)
        #randn(n)返回一个一维数组，具有标准正态分布。
        wordEmbedding.append(np.random.randn(self._embeddingSize)) 
        #形如：[-1.23724451e+00, -2.07029303e-02, -4.16660965e-04, -6.14954146e-01,...],此处维度为200
        #print ("wordEmbedding:",wordEmbedding)

        #words:删除低频词后的词汇表
        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + " not in wordvec")

        #print ("vocab_wordVec:",vocab)
        print ("wordEmbedding:",len(wordEmbedding))
        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """

        print ("_readStopWord():",5)
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            #print("stopWordList:",stopWordList)  #形如：['is', 'a', 'at']
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
            #print ("self.stopWordDict:",self.stopWordDict)  #形如：{'is': 0, 'a': 1, 'at': 2}
            #print ('\n')
        print ("self.stopWordDict:",self.stopWordDict)

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化停用词
        print ("begin creating wordEmbedding...",4)
        self._readStopWord(self._stopWordSource)

        # 初始化数据集,得到分词后的句子和标签
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵
        word2idx,label2idx = self._genVocabulary(reviews, labels)

        # 将标签和句子数值化
        print ("labels:",type(labels),labels)
        print ("label2idx:",type(label2idx),label2idx)
        labelIds = self._labelToIndex(labels, label2idx)   #索引表
        reviewIds = self._wordToIndex(reviews, word2idx)  #索引表

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds, labelIds, word2idx,
                                                                                    self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

        print ("wordEmbedding has been created...'\n")


data = Dataset(config)
data.dataGen()

#输出batch数据集
def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    #x:trainReviews, y:trainLabels, config.batchSize
    # trainReviews:以单词索引构成的句子。
    # trainlabels: 以标签索引构成。
    # arange(x):返回一个序列（向量），形如：0,1,2,...x-1
    perm = np.arange(len(x))
    # shuffle(x)：将数组x随机打乱，改变自身序列。无返回值。
    np.random.shuffle(perm)
    # 打乱训练集和训练标签（句子与标签对应）
    x = x[perm]
    y = y[perm]

    # /表示浮点数除法，返回浮点结果。//表示整数除法，返回一个不大于结果的最大整数。
    # numBatches: iteration次数
    numBatches = len(x) // batchSize    #这里就是15 1022/64

    # batchX存放数据：
    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")
        #print ("i and batchX:",i,len(batchX))
        # yield:类似一个return，返回本次batchX和batchY。并且记住这个返回的位置（i值），下次迭代就从这个位置后开始。
        yield batchX, batchY


# 构建模型：输入，词嵌入层，卷积层，池化层，全连接层
class TextCNN(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        #print ("text_CNN():begin",2)
        # placeholder(dtype,shape=None,name=None)。
        """ 作用：在神经网络构建graph时在模型中占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
        等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        参考：https://blog.csdn.net/kdongyi/article/details/82343712 """
        # dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
        # shape：默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
        # name:名称
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")#防止过拟合

        # 定义l2损失
        # tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)
        # 创建常量。value:必须提供。使用eval()函数查看创建的值。
        # shape: 张量的形状，即维数以及每一维的大小。如果指定了shape
        # 当第一个参数value是数字时，张量的所有元素都会用该数字填充。
        # 而当第一个参数value是一个列表时，注意列表的长度必须小于等于第三个参数shape的大小（即各维大小的乘积）
        # verify_shape默认为False，如果修改为True的话表示检查value的形状与shape是否相符
        l2Loss = tf.constant(0.0)

        # 词嵌入层，tf.name_scope("embedding")，定义一块名为embedding的区域，并在其中工作。参考：https://www.jianshu.com/p/635d95b34e14
        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵，tf.Variable(initializer,name,trainable=True),initializer:初始化参数。
            # trainable:是否把变量添加到collection GraphKeys.TRAINABLE_VARIABLES 中，（
            # collection 是一种全局存储，不受变量名生存空间影响，一处保存，到处可取）
            # tf.cast(x, dtype, name=None),类型转换函数。x:输入，dtype：转换目标类型。返回：Tensor
            # print ("wordEmbedding:",len(wordEmbedding))
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            # tf.nn.embedding_lookup(params, ids):目的是按照ids从params这个矩阵中拿向量（行），
            # 所以ids就是这个矩阵索引（行号），需要int类型。
            # params：完整的嵌入张量，或者除了第一维度之外具有相同形状的P个张量的列表，表示经分割的嵌入张量
            # 一个类型为int32或int64的Tensor，包含要在params中查找的id。参考：https://blog.csdn.net/yangfengling1023/article/details/82910951
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            # tf.expand_dims(input,axis=None,name=None,dim=None)。给定一个input,在axis轴处给input增加一个为1的维度。
            # 例：原始矩阵shape[2,3,5].axis=0,shape变为[1,2,3,5].axis=2,shape变为[2,3,1,5].axis=-1,shape变为[2,3,5,1]
            #参考：https://blog.csdn.net/qq_20014079/article/details/82804374
            self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)
            #print ("self.W:",self.W)
            #print ("self.inputX:",self.inputX)
            #print ("self.embeddedWords:",self.embeddedWords)
            #print ("self.embeddedWordsExpanded:",self.embeddedWordsExpanded)

        # 创建卷积和池化层
        pooledOutputs = []
        # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        # config.model.filterSizes:[3,4,5],embeddingSize:200,numFilters:128
        for i, filterSize in enumerate(config.model.filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                # 卷积层尺寸：filterSize * embeddingSize * 1 * numFilters
                filterShape = [filterSize, config.model.embeddingSize, 1, config.model.numFilters]
                # tf.truncated_normal(shape,mean,stddev,dtype,seed,name):产生截断正态分布随机数。
                # shape:输出张量的维度。stddev:标准差。该函数产生的随机数与均值的差距不会超过两倍的标准差。
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                # 创建一个shape为numFilters的，值为0.1的常量张量。
                b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
                # tf.nn.conv2d(input,filter,strides,padding):实现卷积操作。
                # input: 需要做卷积的嵌入向量，要求是一个4维的Tensor，要求类型为float32或64。
                # shape为：[batch,in_height,in_width,in_channels]
                # filter:卷积核，要求是一个Tensor。shape为：[filterSize, embeddingSize, in_channels, numFilters]
                # strides:卷积时的步长。是一个一维向量，长度为4。padding:"SAME"(边缘填充)或"VALID"（边缘不填充），卷积方式。
                # 返回一个Tensor,即feature map。 返回shape为：[batch,height,width,out_channels]。
                # 返回的height:对一个长度为n的句子，以高度为h的卷积核进行操作，得到n-h+1=height。
                # 返回的width: 宽度为d1的单词，以宽度为d2的卷积核进行操作，得到n-h+1=width
                # 返回的out_channels:即卷积核个数。
                # 参考：https://zhuanlan.zhihu.com/p/26139876
                conv = tf.nn.conv2d(
                    self.embeddedWordsExpanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                #print ("embeddedWordsExpanded:",self.embeddedWordsExpanded)
                #print ("W:",W)
                #print ("b:",b)
                #print ("conv:",conv)

                # relu函数的非线性映射
                # relu(features,name=None):计算校正线性。将大于0的数保持不变，小于0的数置为0.
                # features:一个Tensor。返回：一个Tensor，与features具有相同的类型。
                # bias_add(value,bias):将bias添加到value的每一行。vaule:一个Tensor。bias，一个一维Tensor，大小与value的最后一个维度匹配。返回：与value具有相同类型的Tensor.
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                # tf.nn.max_pool(value,ksize,strides,padding,name=None)
                # value: 需要池化的输入。一般池化层接在卷积层后，所以输入通常是feature map。shape为：[batch,height_1,width_1,channels]
                # ksize:池化窗口的大小。shape一般为：[1,height_2,width_2,1]。一般来说：height_1=hwight_2,width_1=width_2
                #strides：窗口在每一个维度上滑动的步长。padding：填充方式。
                #返回一个Tensor.shape为:[batch,height_1-height_2+1,width_1-width_2+1,channels]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.sequenceLength - filterSize + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print ("pooled:",pooled)
                pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中
        #print ("pooledOutputs:",pooledOutputs)

        # 得到CNN网络的输出长度
        numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)
        #print ("numFiltersTotal:",numFiltersTotal)

        # 池化后的维度不变，按照最后的维度channel来concat
        # concat([tensor1,tensor2,...],axis):用于拼接张量。axis,拼接的维度。
        self.hPool = tf.concat(pooledOutputs, 3)
        #print ("self.hPool:",self.hPool)

        # 摊平成二维的数据输入到全连接层
        # reshape(tensor,shape,name):重塑张量。如果shape的一个分量为-1，则计算该维度的大小，以保持总大小不变。至多有一个分量为-1.
        # 参考：https://www.w3cschool.cn/tensorflow_python/tensorflow_python-bumd2ide.html
        self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])
        #print ("self.hPoolFlat:",self.hPoolFlat)

        # dropout(x,keep_prob,noise_shape=None,seed=None):用于减轻或防止过拟合。
        # Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
        # x:输入Tensor。keep_prob:每个元素被保留下来的概率。在初始化时，为一个占位符。tensorflow在run时，设置keep_prob具体的值。
        # train的时候才是dropout起作用的时候，test的时候不应该让dropout起作用
        # 返回：与x相同形状的张量。
        with tf.name_scope("dropout"):
            self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)
        #print ("self.hDrop:",self.hDrop,'\n')

        # 全连接层的输出
        # tf.get_variable(name,shape,initializer): 作用是创建新的shape形式的tensor变量。
        # xavier_initializer()：返回初始化权重矩阵。
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[numFiltersTotal, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            #print ("outputW:",outputW)
            #print ("outputB:",outputB)
            # loss:用于测量两个张量之间或者张量与0之间的误差。
            # l2_loss(t,name):计算L2损失。简单的可以理解成张量中的每一个元素进行平方,然后求和,最后乘一个1/2.
            # 返回：一个与t相同类型的tensor。
            l2Loss += tf.nn.l2_loss(outputW)
            #print ("l2Loss:",l2Loss)
            l2Loss += tf.nn.l2_loss(outputB)
            #print ("l2Loss:",l2Loss)
            # tf.nn.xw_plus_b(x,weights,biases,name):计算x*weights+biases.
            # x：2d Tensor.shape为:[batch,in_units]。
            # weight:2d tensor.x的权重矩阵，一般都是可训练的。shape为：[in_units,out_units].
            # biases:1d tensor.shape:out_units
            # 返回：2d tensor。shape为：[bath, out_units]
            self.logits = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="logits")
            #print ("self.logits:",self.logits)
            # 二分类
            # tf.greater_equal(x,y):x,y形状相同。如果x比y相同位置的值大,返回True。否则返回False
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
            # 多分类
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

            #print("self.predictions:",self.predictions)

        # 计算二元交叉熵损失
        # tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None,labels=None,logists=None,name):计算给定logits的sigmoid交叉熵。
        # 测量离散分类任务中的概率误差。
        # logists与labels数据类型相同，shape相同。
        # 返回：与logits相同shape的tensor。是逻辑损失
        with tf.name_scope("loss"):

            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
                #print ("losses:",losses)
                #print ("inputY:",tf.reshape(self.inputY, [-1, 1]))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            # reduce_mean(input_tensor,axis=None,keep_dims=False,name,reduction_indices=None):计算张量各个维度上元素的平均值。
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss
            #print ("self.loss:",self.loss)


"""
定义各类性能指标
"""

def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def binary_precision(pred_y, true_y, positive=1):
    """
    二类的精确率计算
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    """
    二类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    """
    二类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param beta: beta值
    :param positive: 正例的索引表示
    :return:
    """
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b


def multi_precision(pred_y, true_y, labels):
    """
    多类的精确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    """
    多类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec


def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    """
    多类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :param beta: beta值
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta


def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    得到二分类的性能指标
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    return acc, recall, precision, f_beta


def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    """
    得到多分类的性能指标
    :param pred_y:
    :param true_y:
    :param labels:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta

# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding   #删除停用词以及低频词后的，其他单词的词向量。
labelList = data.labelList  #[0,1]

# 定义计算图
"""tf.Graph() 表示实例化了一个类，一个用于 tensorflow 计算和表示用的数据流图，通俗来讲就是：在代码中添加的操作（画中的结点）和
数据（画中的线条）都是画在纸上的“画”，而图就是呈现这些画的纸，你可以利用很多线程生成很多张图，但是默认图就只有一张。
tf.Graph().as_default() 表示将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图，
如果只有一个主线程不写也没有关系，tensorflow 里面已经存好了一张默认图，可以使用tf.get_default_graph() 来调用（显示这张默认纸），
当你有多个线程就可以创造多个tf.Graph()，就是你可以有一个画图本，有很多张图纸，这时候就会有一个默认图的概念了。"""
with tf.Graph().as_default():
    print ("text_CNN():begin")
    #tf.ConfigProto配置tf.Session的运算方式，比如GPU运算或者CPU运算。
    #allow_soft_placement=True：有时候，不同的设备，它的cpu和gpu是不同的，如果将这个选项设置成True，那么当运行设备不满足要求时，会自动分配GPU或者CPU。
    #log_device_placement=True:打印Tensorflow使用了哪种操作。
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True  #当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    #Session(config),Session对象封装了Operation执行对象的环境，并对Tensor对象进行计算
    #congig，session的各种配置选项。
    sess = tf.Session(config=session_conf)

    # 定义会话，sess.as_default(),返回默认session的上下文管理器
    with sess.as_default():
        # 计算训练模型所需的时间
        TrainbeginTime = datetime.datetime.now()
        cnn = TextCNN(config, wordEmbedding)
        
        #print ("cnn:",cnn)
        print ("textcnn create over...")

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        #print ("globalStep:",globalStep)
        # 定义优化函数，传入学习速率参数
        # tf.train.AdamOptimizer(learning_rate):Adam优化算法，寻找一个全局最优的优化算法。
        # 传统的梯度下降算法中的学习率不会改变，Adman算法的通过计算梯度的一阶矩估计核二阶据估计，为不同的参数设计独立的学习率
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量。optimizer.compute_gradients(loss)：计算loss的梯度。返回一个以元组(gradient,variable)组成的列表。
        # 梯度下降。参考：https://www.zhihu.com/question/305638940/answer/670034343，https://zhuanlan.zhihu.com/p/43452377
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        #print ("gradsAndVars:",gradsAndVars)
        # 将梯度应用到变量下，生成训练器.apply_gradients(grads),将计算出的梯度应用到变量上，用得到的gradient来更新对应的variable
        # 返回一个指定的梯度的操作Operation，对gobalstep做自增操作。
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        #print ("trainOp:",trainOp)

        # 用summary绘制tensorBoard
        # 可视化方法：https://www.pianshen.com/article/8287418009/
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                # tf.summary.histogram(tags,values):用来显示直方图信息，显示训练过程中变量的分布情况。
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                # scalar(tags,values):显示标量信息，在画loss,accuary时使用。
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #abspath:获得绝对路径。join(dirname,basename):将文件路径和文件名凑成完整文件路径
        outDir = os.path.abspath(os.path.join("../", "summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", cnn.loss)
        #merge_all()：自动管理。该模式下，导入已保存的模型继续训练时，会抛出异常。
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        print("moooodel")
        # tf.summary.FileWriter(path,sess.graph):指定一个文件来保存图。
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        # tf.global_variables()：返回所有变量，也可以指定scopr参数查看指定域的变量
        # tf.train.Saver()：保存模型。创建Saver对象时，max_to_keep参数表示要保留最近的5个模型。
        # 如果每训练一代(epoch)就保存一次模型，则设置为max_to_keep=0。为1时表示保存最后一次模型。
        # 使用saver.save()方法保存训练好的模型。
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        """savedModelPath = "../savedModel"
        #判断文件是否存在
        if os.path.exists(savedModelPath):
            #删除空目录
            os.rmdir(savedModelPath)
        # 准备存储模型
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)"""
        # tf.global_variables_initializer():真正为全局变量赋值
        sess.run(tf.global_variables_initializer())


        def trainStep(batchX, batchY):
            """
            训练函数
            """

            # batchX: 由单词索引构成的句子。句子长度为200，batchX大小为：batchsize
            # batchY：batchX中句子对应的标签。
            # tensorflow中，a=tf.placeholder()为占位符。feed_dict={a:value,b:value},以字典形式赋值。
            #print ("batchY:",batchY)
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: config.model.dropoutKeepProb
            }

            # trainOp:指定的梯度的操作Operation。summaryOp：自动管理。globalStep：执行步骤，trainOp会对其进行自增。
            # cnn.loss：批量数据的损失值（一位数）。cnn.predictions：批量数据的预测结果，shape为:[batchsize,1]。
            _, summary, step, loss, predictions = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions],
                feed_dict)
            # 日期。
            timeStr = datetime.datetime.now().isoformat()

            #print ("type_predictions_batchY:",type(predictions),type(batchY))
            if config.numClasses == 1:
                #根据批量数据的预测值和真实值，计算指标。
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                              labels=labelList)

            # 在文件中保存图
            trainSummaryWriter.add_summary(summary, step)

            return loss, acc, recall, prec, f_beta


        def devStep(batchX, batchY):
            """
            验证函数
            """

            # 验证集。在验证集中，所有神经元处于活跃状态。
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions],
                feed_dict)

            if config.numClasses == 1:
                acc, recall, precision, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
            elif config.numClasses > 1:
                acc, recall, precision, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, recall, precision, f_beta

        

        # 1个epoch=iteration数 * batchsize数=训练集中的全部样本训练一次。
        # 使用多次epoch，将完整的数据集在同样的神经网络中传递多次。不断更新权重矩阵。使得曲线从欠拟合到过拟合。
        # 1个epoch=所有训练样本的‘一个正向传递和一个反向传递’=使用训练集中的全部样本训练一次
        # 1个iteration = 1个正向通过+1个反向通过=使用batchsize个样本训练一次。每一次迭代得到的结果都会被作为下一次迭代的初始值
        # batchsize: 批量大小，即每次村连的样本数目。Batch_Size的正确选择是为了在内存效率和内存容量之间寻找最佳平衡
        # 参考：https://zhuanlan.zhihu.com/p/78178208，https://blog.csdn.net/program_developer/article/details/78597738
        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            # nextBatch返回批量的数据x和y
            # x:由单词索引构成的批量句子(大小为batchSize)。y：x对应的标签索引
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                #print ("batchTrainX:",len(batchTrain[0]),batchTrain[0], batchTrain[1])
                # 计算训练集中，批量数据的指标。
                loss, acc, recall, prec, f_beta = trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))
                # 此处表示，当使用训练集完成一轮训练时（所有训练集训练一次）。执行if，使用验证集评估模型性能。
                if currentStep % config.training.evaluateEvery == 0:
                    print(currentStep)
                    print("\nEvaluation:")

                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []

                    # 将验证集分批处理。对每次处理得到的loss，acc，pre，recall，f1相加。最后将所有批次的结果求均值。
                    # 以此验证模型性能。
                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, recall, precision, f_beta = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accs),
                                                                                                         mean(
                                                                                                             precisions),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)))

                # 每完成一次epoch，保存模型。
                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    # save(sess,path,global_step):保存模型。path:路径和名字。global_step:将训练次数作为后缀加到模型名字中
                    # 保存了一个文件为 checkpoint的 文件，保存了一个目录下所有的模型文件列表。
                    # 该模型在训练集中使用。
                    # meta文件：保存神经网络的网络结构。 data文件：数据文件，保存神经网络的权重，偏置，操作等。index文件：字符串表，每一个键都是张量的名字。值描述张量的元数据。
                    # checkpoint文件：文本文件，记录了训练过程中，在所有中间节点上保存的模型的名称。首行记录的时最后一次保存的模型名称。
                    # saver()中的参数使得保存最近的5个模型
                    path = saver.save(sess, "../model/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))
        TrainendTime = datetime.datetime.now()
        # tf.saved_model.utils.build_tensor_info 是把变量变成可缓存对象的函数
        # inputs:输入
        """inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}
        # outputs 模型预测的结果
        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(cnn.predictions)}

        # 然后把两个字典打包放入 signature 中
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        # tf.tables_initializer()：返回默认图表的所有表的操作。  tf.group()：用于创造一个操作，可以将传入参数的所有操作进行分组。
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        # 导入graph信息以及变量
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()"""



# 将训练好的模型用于预测
def review_to_wordlist(review):
    # 把评论转成词序列
    # 参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613

    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words


#计算预测时间

TestbeginTime = datetime.datetime.now()
# 测试集
test = pd.read_excel('E:/学习/梁老师实验室/王田路需求分类/非功能需求自动分类（基于系统模型）/实验数据/test_data1.xlsx', header=0, delimiter="\t")
labels = test['category'].tolist()
test_data = []
# 分词后的句子
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# print(test_data)

# x = "it really annoys me when random contacts try to video call me"

# 预测代码
# 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
# 词汇-索引映射表
with open("word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)
# 类型-索引映射表
with open("label2idx.json", "r", encoding="utf-8") as f:
    label2idx = json.load(f)
# label2idx: {'P': 0, 'N': 1}
idx2label = {value: key for key, value in label2idx.items()}  # {0:'P',1:'N'}

#print ("label2idx_idx2label:",label2idx,idx2label)
"""
for key,value in label2idx.items():
    value:key
"""

pre_final = []
#np.toarray(pre_final)
# 处理测试集中的每个句子
for x in test_data:
    # 句子由单词表示转换为索引表示
    xIds = [word2idx.get(item, word2idx["UNK"]) for item in x.split(" ")]
    # 填充句子长度
    if len(xIds) >= config.sequenceLength:
        xIds = xIds[:config.sequenceLength]
    else:
        xIds = xIds + [word2idx["PAD"]] * (config.sequenceLength - len(xIds))

    graph = tf.Graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # checkpointfile:二进制文件。包含了所有权重，偏置，梯度和其他变量的值。
            # tf.train.latest_checkpoint：自动寻找最近保存的文件
            checkpoint_file = tf.train.latest_checkpoint("../model/")
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            # restore():将训练好的模型提取出来。
            saver.restore(sess, checkpoint_file)

            # 获得需要喂给模型的参数，输出的结果依赖的输入值
            inputX = graph.get_operation_by_name("inputX").outputs[0]
            dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

            # 获得输出的结果
            predictions = graph.get_tensor_by_name("output/predictions:0")

            pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]

    pred = [idx2label[item] for item in pred]
    # ''.join(list):将list转为str
    pre_final.append(''.join(pred))
    #print(pred)


#pre_final由字符串转换为0，1表示的数组。
for i in range(len(pre_final)):
    pre_final[i] = int(pre_final[i])

print ("textcnn_word2Vec:")
"""print(accuracy_score(labels, pre_final),
      precision_score(labels, pre_final),
      recall_score(labels, pre_final),
      f1_score(labels, pre_final))"""
print ('\n')

accuracy_final,recall_final, precision_final,f1_final = get_binary_metrics(np.array(pre_final),np.array(labels))
print ("accuracy_final,recall_final,precision_final,f1_final:",accuracy_final,recall_final,precision_final,f1_final)

endTime = datetime.datetime.now()

#计算textCNN_word2Vec，处理词向量，构造模型，训练模型，验证模型，测试数据整个流程所用的时间

print ("endtime-begintime:",((endTime - beginTime).seconds * 1000 + (endTime - beginTime).microseconds / 1000),(endTime-beginTime).seconds)


print ("TrainendTime-TrainbeginTime:",((TrainendTime - TrainbeginTime).seconds * 1000 + (TrainendTime - TrainbeginTime).microseconds / 1000),(TrainendTime - TrainbeginTime).seconds)

print ("testTime:",((endTime - TestbeginTime).seconds * 1000 + (endTime - TestbeginTime).microseconds / 1000),(endTime - TestbeginTime).seconds)







