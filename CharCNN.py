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
import warnings
warnings.filterwarnings("ignore")

#设置系统打印信息：为2时，打印Error和Fatal信息。屏蔽info和warning信息。
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


beginTime = datetime.datetime.now()

#配置参数
class TrainingConfig(object):
    print ("TrainingConfig():",1)
    epoches = 10
    evaluateEvery = 15       #训练集样本数1022 // batchsize = 15。每完成一次训练，进行评估。
    checkpointEvery = 15     #每100轮保存模型
    learningRate = 0.001

class ModelConfig(object):
    print ("ModelConfig():",2)
    # 定义6个卷积层。该列表中子列表的三个元素分别是卷积核的数量，卷积核的高度，池化的尺寸
    convLayers = [[256,7,3],[256,7,3],
                    [256,3,None],[256,3,None],[256,3,None],
                    [256,3,3]]
    fcLayers = [1024,1024]
    dropoutKeepProb = 0.5

    epsilon = 1e-3   #BN层中防止分母为0而加入的极小值
    decay = 0.999    #BN层中用来计算滑动平均的值
    

class Config(object):
    print ("config():",3)
    #使用论文中提出的69个字符来表征输入数据
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

    sequenceLength = 1014   #字符表示的序列长度
    batchSize = 64

    dataSource = "dataSet0320.xlsx"
    #stopWordSource = "StopWords.txt"

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
        #self._stopWordSource = config.stopWordSource #停用词

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._alphabet = config.alphabet
        self.charEmbedding = None #词向量维度
        #self._batchSize = config.batchSize              
        self._rate = config.rate                   #训练集比例

        #self._stopWordDict = {}                   #在_readStopWord()函数中被赋值

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self._charToIndex = {}
        self._indexToChar = {}
        #self._labelToIndex = {}
        #self.labelList  = {}

        #self.wordEmbedding = None              #在_genVocabulary()函数中被赋值。存储：删除停用词以及低频词后的，其他单词的词向量

        #self.labelList = []                    #在_genVocabulary()函数中被赋值。存储：[0,1]

    def _readData(self, filePath):
        """
        从excel文件中读取数据集
        """
        print ("_readData():",5)
        df = pd.read_excel(filePath)

        
        labels = df["category"].tolist()
        review = df["review"].tolist()
        #print ("labels:",labels)
        #将句子用字符表示
        reviews = [[char for char in line if char !=" "] for line in review]
        #print ("reviews:",reviews)
        #返回所有词以及句子标签
        return reviews, labels

    def _reviewProcess(self, review, sequenceLength, charToIndex):
        """
        将数据集中的每条评论用索引表示
        wordToIndex中“pad”对应的Index为0
        """
        
        # 句子中的字符长度1024
        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength
        
        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)

        # 用字符索引表示句子
        for i in range(sequenceLen):
            if(review[i] in charToIndex):
                reviewVec[i] = charToIndex[review[i]]
            else:
                reviewVec[i] = charToIndex["UNK"]
        
        return reviewVec


    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """
        #x:以单词索引构成的dict，y:以标签索引构成的数组，word2idx:词汇-索引表，rate:数据分割
        print ("_genTrainEvalData():step",8,'\n')
        #存放由单词索引构成的句子。每个句子长度为200
        reviews = []
        labels = []

        print ("_reviewProcess():step",9,'\n')
        #遍历所有文本，将文本中的词转换为index表示
        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i],self._sequenceLength,self._charToIndex)
            reviews.append(reviewVec)
            labels.append([y[i]])

        #print ("reviewVec:",reviewVec)
        # 用字符索引表示的句子向量。每个句子长度为1024
        #print ("reviews:",reviews)
        #print ("labels:",labels)

        #训练集大小：原始数据集大小*rate
        trainIndex = int(len(x) * rate)
        #a[n:],截取从n后的所有元素，包含n位置。a[:n],截取从0开始到n-1的所有元素，不包含n位置。
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        print("trainReviews:",len(trainReviews),trainReviews)

        trainLabels = np.array(labels[:trainIndex], dtype="float32")
        print ("trainLabels:",trainLabels)

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        #print("evalReviews:",len(evalReviews),evalReviews)
        evalLabels = np.array(labels[trainIndex:], dtype="float32")
        # trainReviews:以字符索引构成的句子。
        # trainlabels: 以标签索引构成。
        return trainReviews, trainLabels, evalReviews, evalLabels

    # 生成字符向量和字符-索引映射字典
    def _genVocabulary(self, reviews):
        print ("_genVocabulary():",6)

        chars = [char for char in self._alphabet]
        print ("chars:",chars,'\n')

        vocab, charEmbedding = self._getCharEmbedding(chars)
        self.charEmbedding = charEmbedding

        #print ("charEmbedding:",self.charEmbedding,'\n')   #[[0,0,0,..0],[]]
        print ("get vocab_charVec and charEmbedding...over")

        #字符-索引
        self._charToIndex = dict(zip(vocab, list(range(len(vocab)))))
        #print ("charToIndex:",self._charToIndex)
        
        #索引-字符
        self._indexToChar = dict(zip(list(range(len(vocab))),vocab))
        #print ("indexToChar:",self._indexToChar)

        
        # 标签-索引
        #self._labelToIndex = label2idx



        # 将字符-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("charToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._charToIndex, f) #将转换完格式后的字符串写入文件
        #print(word2idx)

        with open("indexToChar.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToChar, f)

        
        print ('\n')
        #return label2idx

    def _getCharEmbedding(self, chars):
        """
        按照one的形式将字符映射成向量
        """
        print ("_genVocabulary():",7)

        alphabet = ["UNK"] + [char for char in self._alphabet]
        vocab = ["pad"] + alphabet
        print ("alphabet:",alphabet)
        print ("vocab:",vocab)
        charEmbedding = []
        # 插入一个全0向量
        charEmbedding.append(np.zeros(len(alphabet),dtype="float32"))
        #print ("charEmbedding1:",charEmbedding)

        for i,alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet),dtype="float32")

            #生成每个字符对应的向量
            onehot[i] = 1

            #生成字符嵌入的向量矩阵
            charEmbedding.append(onehot)

        #print ("charEmbedding:",charEmbedding)
        #print ("alphabet:",alphabet,'\n')
        #print ("vocab:",vocab) 
        return vocab, np.array(charEmbedding)


    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化数据集
        print ("begin creating charEmbedding...",4)
    
        # 初始化数据集,得到分词后的句子和标签
        reviews, labels = self._readData(self._dataSource)
        #print ("reviews,labels:",reviews,labels)

        #初始化字符-索引映射表和词向量矩阵
        self._genVocabulary(reviews)

        """print ("labels:",type(labels),labels)
        print ("label2idx:",type(label2idx),label2idx)
        labelIds = self._labelToIndex(labels,label2idx)"""


        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

        print ("charEmbedding has been created...'\n")


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


# 构建模型
class charCNN(object):
    """
    charCNN 用于文本分类
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
        self.inputY = tf.placeholder(tf.float32, [None,1], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")#防止过拟合

        #tf.bool布尔类型
        self.isTraining = tf.placeholder(tf.bool,name="isTraining")

        self.epsilon = config.model.epsilon
        self.decay = config.model.decay


        # 字符嵌入，tf.name_scope("embedding")，定义一块名为embedding的区域，并在其中工作。参考：https://www.jianshu.com/p/635d95b34e14
        with tf.name_scope("embedding"):

            # 利用one-hot的字符向量初始化词嵌入矩阵，tf.Variable(initializer,name,trainable=True),initializer:初始化参数。
            # trainable:是否把变量添加到collection GraphKeys.TRAINABLE_VARIABLES 中，（
            # collection 是一种全局存储，不受变量名生存空间影响，一处保存，到处可取）
            # tf.cast(x, dtype, name=None),类型转换函数。x:输入，dtype：转换目标类型。返回：Tensor
            print ("charEmbedding:",len(charEmbedding))
            self.W = tf.Variable(tf.cast(charEmbedding, dtype=tf.float32, name="charEmbedding"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            # tf.nn.embedding_lookup(params, ids):目的是按照ids从params这个矩阵中拿向量（行），
            # 所以ids就是这个矩阵索引（行号），需要int类型。
            # params：完整的嵌入张量，或者除了第一维度之外具有相同形状的P个张量的列表，表示经分割的嵌入张量
            # 一个类型为int32或int64的Tensor，包含要在params中查找的id。参考：https://blog.csdn.net/yangfengling1023/article/details/82910951
            # 获取字符嵌入
            self.embeddedChars = tf.nn.embedding_lookup(self.W, self.inputX)
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            # tf.expand_dims(input,axis=None,name=None,dim=None)。给定一个input,在axis轴处给input增加一个为1的维度。
            # 例：原始矩阵shape[2,3,5].axis=0,shape变为[1,2,3,5].axis=2,shape变为[2,3,1,5].axis=-1,shape变为[2,3,5,1]
            #参考：https://blog.csdn.net/qq_20014079/article/details/82804374
            # 添加一个通道维度
            self.embededCharsExpand = tf.expand_dims(self.embeddedChars, -1)
            print ("self.W:",self.W)
            print ("self.inputX:",self.inputX)
            print ("self.embeddedChars:",self.embeddedChars)
            print ("self.embededCharsExpand:",self.embededCharsExpand)

        # 卷积-池化层
        # config.model.convlayers:[[256,7,4],[256,7,4],[256,3,4]]
        for i, cl in enumerate(config.model.convLayers):
            print ("The "+str(i+1)+" conv address...")

            with tf.name_scope("convLayers-%s" % (i+1)):
                # 获取字符的向量长度
                filterWidth = self.embededCharsExpand.get_shape()[2].value
                print ("filterWidth:",filterWidth) 
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                # 卷积层尺寸：filterSize * embeddingSize * 1 * numFilters
                filterShape = [cl[1],filterWidth,1,cl[0]]
                print ("filterShape:",filterShape) #(7,70,1,256)

                # 初始化卷积核权重参数
                stdv = 1 / sqrt(cl[0] * cl[1])  #1/sqrt(256*7)
                print ("stdv:",stdv)
                # 初始化权重矩阵w和偏置b
                # tf.random_uniform(shape,minval=0,maxval=None):从均匀分布中输出随机值，生成的值在[minval,maxval)范围内遵循均匀分布
                # shape:输出张量的形状。返回：用于填充随机均匀值的指定形状的张量
                wConv = tf.Variable(tf.random_uniform(filterShape,minval = -stdv,maxval = stdv),dtype="float32",name="W")

                bConv = tf.Variable(tf.random_uniform(shape = [cl[0]],minval = -stdv,maxval = stdv),name="b")

                # 高斯分布，初始化.stddev:标准差。
                # wConv = tf.Variable(tf.truncated_normal(filterShape, stddev=0.05),dtype="float32",name="W")

                # bConv = tf.Variable(tf.constant(0,shape = [cl[0]]),name="b")

                # 构建卷积层，可以直接将卷积核的初始化方法传入（w_conv）
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
                    self.embededCharsExpand,
                    wConv,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                #print ("embededCharsExpand:",self.embededCharsExpand)
                print ("wConv:",wConv)
                print ("bConv:",bConv)
                print ("conv:",conv)


                # relu函数的非线性映射
                # relu(features,name=None):计算校正线性。将大于0的数保持不变，小于0的数置为0.
                # features:一个Tensor。返回：一个Tensor，与features具有相同的类型。
                # bias_add(value,bias):将bias添加到value的每一行。vaule:一个Tensor。bias，一个一维Tensor，大小与value的最后一个维度匹配。返回：与value具有相同类型的Tensor.
                hConv = tf.nn.relu(tf.nn.bias_add(conv,bConv))
                print ("hConv:",hConv,'\n')
                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                # tf.nn.max_pool(value,ksize,strides,padding,name=None)
                # value: 需要池化的输入。一般池化层接在卷积层后，所以输入通常是feature map。shape为：[batch,height_1,width_1,channels]
                # ksize:池化窗口的大小。shape一般为：[1,height_2,width_2,1]。一般来说：height_1=hwight_2,width_1=width_2
                # strides：窗口在每一个维度上滑动的步长。padding：填充方式。
                # 返回一个Tensor.shape为:[batch,height_1-height_2+1,width_1-width_2+1,channels]
                if cl[-1] is not None:
                    ksizeShape = [1,cl[-1],1,1]
                    hPool = tf.nn.max_pool(
                        hConv,
                        ksize=ksizeShape,
                        # ksize shape: [batch, height, width, channels]
                        strides=ksizeShape,
                        padding="VALID",
                        name="pool")
                else:
                    hPool = hConv

                print ("hPool.shape:",hPool.shape)

                # 对维度进行转换，转换成卷积层的输入维度
                # tf.transpose(a,perm=None,name):置换a，根据perm重新排列尺寸。返回：张量维度i将对应输入维度perm[i]
                self.embededCharsExpand = tf.transpose(hPool,[0,1,3,2],name = "transpose")
                print ("self.embededCharsExpand:",self.embededCharsExpand)

        print ("self.embededCharsExpand:",self.embededCharsExpand)

        with tf.name_scope("reshape"):
            fcDim = self.embededCharsExpand.get_shape()[1].value * self.embededCharsExpand.get_shape()[2].value
            print ("fcDim:",fcDim)
            # 摊平成二维的数据输入到全连接层
            # reshape(tensor,shape,name):重塑张量。如果shape的一个分量为-1，则计算该维度的大小，以保持总大小不变。至多有一个分量为-1.
            # 参考：https://www.w3cschool.cn/tensorflow_python/tensorflow_python-bumd2ide.html
            self.inputReshape = tf.reshape(self.embededCharsExpand,[-1,fcDim])
            print ("inputReshape:",self.inputReshape)

        # 全连接层
        weights = [fcDim] + list(config.model.fcLayers)
        print ("weights:",weights)

        for i, fl in enumerate(config.model.fcLayers):
            with tf.name_scope("fcLayers-%s"%(i+1)):
                print ("The "+ str(i+1) + " softmax address...")
                stdv = 1 / sqrt(weights[i])

                print ("fcLayers",fl)
                #定义全连接层的初始化方法，均匀分布初始化w和b值
                wFc = tf.Variable(tf.random_uniform([weights[i],fl],minval = -stdv,maxval = stdv),dtype="float32",name="w")
                bFc = tf.Variable(tf.random_uniform(shape = [fl],minval = -stdv, maxval = stdv),dtype="float32",name="b")

                
                # 高斯分布，初始化.stddev:标准差。
                # wFc = tf.Variable(tf.truncated_normal([weights[i],fl], stddev=0.05),dtype="float32",name="W")
                # bFc = tf.Variable(tf.constant(0,shape = [fl]),name="b")

                # 矩阵相乘：inputReshape*wFc
                self.fcInput = tf.nn.relu(tf.matmul(self.inputReshape,wFc) + bFc)


                # dropout(x,keep_prob,noise_shape=None,seed=None):用于减轻或防止过拟合。
                # Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
                # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
                # x:输入Tensor。keep_prob:每个元素被保留下来的概率。在初始化时，为一个占位符。tensorflow在run时，设置keep_prob具体的值。
                # train的时候才是dropout起作用的时候，test的时候不应该让dropout起作用
                # 返回：与x相同形状的张量。
                with tf.name_scope("dropOut"):
                    self.fcInputDrop = tf.nn.dropout(self.fcInput, self.dropoutKeepProb)

            self.inputReshape = self.fcInputDrop
            print ("wFc:",wFc)
            print ("bFc:",bFc)
            print ("self.fcInput:",self.fcInput)
            print ("self.fcInputDrop",self.fcInputDrop)
            print ("self.inputReshape:",self.inputReshape)

        # 定义隐层到输出层的权重系数和偏差的初始化方法
        # tf.get_variable(name,shape,initializer): 作用是创建新的shape形式的tensor变量。
        # xavier_initializer()：返回初始化权重矩阵。
        with tf.name_scope("outputLayer"):
            # 数组下标为-1，表示数组的最后一行数据
            stdv = 1 / sqrt(weights[-1])

            wOut = tf.Variable(tf.random_uniform([weights[-1],1],minval = -stdv, maxval = stdv),dtype="float32",name="w")
            bOut = tf.Variable(tf.random_uniform(shape=[1],minval = -stdv, maxval = stdv), name="b")

            #
            # 高斯分布，初始化.stddev:标准差。
            # wOut = tf.Variable(tf.truncated_normal([weights[-1],1], stddev=0.05),dtype="float32",name="W")
            # bOut = tf.Variable(tf.constant(0,shape = [1]),name="b")

            # tf.nn.xw_plus_b(x,weights,biases,name):计算x*weights+biases.
            # x：2d Tensor.shape为:[batch,in_units]。
            # weight:2d tensor.x的权重矩阵，一般都是可训练的。shape为：[in_units,out_units].
            # biases:1d tensor.shape:out_units
            # 返回：2d tensor。shape为：[bath, out_units]
            self.predictions = tf.nn.xw_plus_b(self.inputReshape, wOut, bOut, name="predictions")
            # 进行二元分类。tf.greater_equal(x,y):x,y形状相同。如果x比y相同位置的值大,返回True。否则返回False
            # True为1，Fasle为0.
            # tf.cast(x,dtype):将x转换为dtype类型的数据。
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.0), tf.float32, name="binaryPreds")

            print ("wOut:",wOut)
            print ("bOut:",bOut)
            print ("self.predictions:",self.predictions)
            print ("self.binaryPreds:",self.binaryPreds)
            #print("self.binaryPreds:",self.binaryPreds)

        # 计算二元交叉熵损失
        # tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None,labels=None,logists=None,name):计算给定logits的sigmoid交叉熵。
        # 测量离散分类任务中的概率误差。
        # logists与labels数据类型相同，shape相同。
        # 返回：与logits相同shape的tensor。是逻辑损失
        with tf.name_scope("loss"):

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
            print ("losses:",losses)
            #print ("inputY:",inputY)

            # reduce_mean(input_tensor,axis=None,keep_dims=False,name,reduction_indices=None):计算张量各个维度上元素的平均值。
            self.loss = tf.reduce_mean(losses)
            print ("self.loss:",self.loss)

    # BN层代码实现           
    """def _batchNorm(self,x):
        gamma = tf.Variable(tf.ones([x.get_shape()[3].value]))
        beta = tf.Variable(tf.zeros([x.get_shape()[3].value]))

        self.popMean = tf.Variable(tf.zeros([x.get_shape()[3].value]),trainable=False,name="popMean")
        self.popVariance = tf.Variable(tf.ones([x.get_shape()[3].value]),trainable=False,name="popVariance")

        def batchNormTraining():
            # 一定要使用正确的维度确保计算的是每个特征图上的平均值和方差而不是整个网络节点上的统计分布值
            batchMean, batchVariance = tf.nn.moments(x,[0,1,2],keep_dims=false)

            decay = 0.99
            trainMean = tf.assign(self.popMean,self.popMean * self.decay + batchMean*(1-self.decay))
            trainVariance = tf.assign(self.popVariance,self.popVariance*self.decay+batchVariance*(1-self.decay))

            # tf.control_dependencies(control_inputs):此函数指定某些操作执行的依赖关系。返回一个控制依赖的上下文管理器
            # 使用 with 关键字可以让在这个上下文环境中的操作都在 control_inputs 执行
            # 参考：https://www.cnblogs.com/reaptomorrow-flydream/p/9492191.html
            # tf.nn.batch_normalization(x,mean,variance,offset,scale,variance_epsilon,name=None)
            # 批量标准化。x:任意维度的输入Tensor。mean:一个平均Tensor。variance:一份方差Tensor。offset:一个偏移量Tensor。
            # scale: 一个标度Tensor。variance_epsilon：一个小的浮点数,以避免除以0.
            # 返回：标准化，缩放，偏移张量。
            with tf.control_dependencies([trainMean,trainVariance]):
                return tf.nn.batch_normalization(x,batchMean,batchVariance,beta,gamma,self.epsilon)

        def batchNormInference():
            return tf.nn.batch_normalization(x,self.popMean,self.popVariance,beta,gamma,self.epsilon)

        batchNormalizedOutput = tf.cond(self.isTraining,batchNormTraining,batchNormInference)
        return tf.nn.relu(batchNormalizedOutput)"""


"""
定义各类性能指标
"""

def mean(item):
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item)
    return res

def getMetrics(binaryPredY, true_y, f_beta=1.0):
    """
    得到二分类的性能指标
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    acc = accuracy_score(true_y, binaryPredY)
    recall = recall_score(true_y,binaryPredY)
    precision = precision_score(true_y,binaryPredY)
    f1 = f1_score(true_y,binaryPredY)

    """acc = accuracy(binaryPredY, true_y)
    recall = binary_recall(binaryPredY, true_y)
    precision = binary_precision(binaryPredY, true_y)
    f1 = binary_f_beta(binaryPredY, true_y, f_beta)"""

    return acc, recall, precision, f1

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


# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

charEmbedding = data.charEmbedding   #字符向量。

# 定义计算图
"""tf.Graph() 表示实例化了一个类，一个用于 tensorflow 计算和表示用的数据流图，通俗来讲就是：在代码中添加的操作（画中的结点）和
数据（画中的线条）都是画在纸上的“画”，而图就是呈现这些画的纸，你可以利用很多线程生成很多张图，但是默认图就只有一张。
tf.Graph().as_default() 表示将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图，
如果只有一个主线程不写也没有关系，tensorflow 里面已经存好了一张默认图，可以使用tf.get_default_graph() 来调用（显示这张默认纸），
当你有多个线程就可以创造多个tf.Graph()，就是你可以有一个画图本，有很多张图纸，这时候就会有一个默认图的概念了。"""
with tf.Graph().as_default():
    print ("char_CNN():begin")
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
        cnn = charCNN(config, charEmbedding)
        #print ("cnn:",cnn)
        print ("charcnn create over...")

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
        outDir = os.path.abspath(os.path.join("../", "charCNN/summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("trainLoss", cnn.loss)
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
        # 准备存储模型
        builder = tf.saved_model.builder.SavedModelBuilder("../charCNN/saveModel")
        # tf.global_variables_initializer():真正为全局变量赋值
        sess.run(tf.global_variables_initializer())


        def trainStep(batchX, batchY):
            """
            训练函数
            """

            # batchX: 由字符索引构成的句子。句子长度为1014，batchX大小为：batchsize
            # batchY：batchX中句子对应的标签。
            # tensorflow中，a=tf.placeholder()为占位符。feed_dict={a:value,b:value},以字典形式赋值。
            #print ("batchY:",batchY)
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: config.model.dropoutKeepProb,
                cnn.isTraining: True
            }

            # trainOp:指定的梯度的操作Operation。summaryOp：自动管理。globalStep：执行步骤，trainOp会对其进行自增。
            # cnn.loss：批量数据的损失值（一位数）。cnn.predictions：批量数据的预测结果，shape为:[batchsize,1]。
            _, summary, step, loss, predictions,binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions,cnn.binaryPreds],
                feed_dict)
            # 日期。
            timeStr = datetime.datetime.now().isoformat()

            #print ("type_predictions_batchY:",type(predictions),type(batchY))

            acc, recall, prec, f1 = getMetrics(binaryPreds,batchY)
            print("{}, step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f1: {}".format
                (timeStr, step, loss, acc, recall, prec,f1))


            # 在文件中保存图
            trainSummaryWriter.add_summary(summary, step)

            #return loss, acc, prec, recall, f_beta


        def devStep(batchX, batchY):
            """
            验证函数
            """

            # 验证集。在验证集中，所有神经元处于活跃状态。
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: 1.0,
                cnn.isTraining: False
            }
            summary, step, loss, predictions,binaryPreds = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions,cnn.binaryPreds],
                feed_dict)


            acc, recall, precision, f1 = getMetrics(binaryPreds,batchY)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, recall,precision, f1

        # 计算训练模型所需的时间
        TrainbeginTime = datetime.datetime.now()
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
            # x:由字符索引构成的批量句子(大小为batchSize)。y：x对应的标签索引
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                #print ("batchTrainX:",len(batchTrain[0]),batchTrain[0], batchTrain[1])
                # 训练模型，批量数据的指标。
                trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
               
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
                    path = saver.save(sess, "../model/charCNN/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))
        TrainendTime = datetime.datetime.now()

        # tf.saved_model.utils.build_tensor_info 是把变量变成可缓存对象的函数
        # inputs:输入
        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}
        # outputs 模型预测的结果
        outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(cnn.binaryPreds)}

        # 然后把两个字典打包放入 signature 中
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        # tf.tables_initializer()：返回默认图表的所有表的操作。  tf.group()：用于创造一个操作，可以将传入参数的所有操作进行分组。
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        # 导入graph信息以及变量
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()


#计算预测时间

TestbeginTime = datetime.datetime.now()
# 测试集
test = pd.read_excel('E:/学习/梁老师实验室/王田路需求分类/非功能需求自动分类（基于系统模型）/实验数据/test_data1.xlsx', header=0, delimiter="\t")
labels = test['category'].tolist()
test_data = []
# 变为字符后的句子
for i in range(len(test['review'])):
    revie = []
    for char in test['review'][i]:
        if char != " ":
            revie.append(char)
    test_data.append(revie)
#print ("test_data:",test_data)

# print(test_data)


# 预测代码
# 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
# 字符-索引映射表
with open("charToIndex.json", "r", encoding="utf-8") as f:
    charToIndex = json.load(f)
print ("charToIndex",charToIndex)

pre_final = []
#np.toarray(pre_final)
# 处理测试集中的每个句子
for x in test_data:
    # 句子由字符表示转换为索引表示
    # charToIndex["UNK"]为1，charToIndex["pad"]为0
    xIds = [charToIndex.get(item, charToIndex["UNK"]) for item in x]
    # 填充句子长度
    if len(xIds) >= config.sequenceLength:
        xIds = xIds[:config.sequenceLength]
    else:
        xIds = xIds + [charToIndex["pad"]] * (config.sequenceLength - len(xIds))

    graph = tf.Graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # checkpointfile:二进制文件。包含了所有权重，偏置，梯度和其他变量的值。
            # tf.train.latest_checkpoint：自动寻找最近保存的文件
            checkpoint_file = tf.train.latest_checkpoint("../model/charCNN/")
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            # restore():将训练好的模型提取出来。
            saver.restore(sess, checkpoint_file)

            # 获得需要喂给模型的参数，输出的结果依赖的输入值
            inputX = graph.get_operation_by_name("inputX").outputs[0]
            #print ("inputX:",inputX)
            dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

            # 获得输出的结果
            binaryPreds = graph.get_tensor_by_name("outputLayer/binaryPreds:0")
            #print("binaryPreds:",binaryPreds)
            pred = sess.run(binaryPreds, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]
            #print("pred:",type(pred),pred)
    #pred = [idx2label[item] for item in pred]
    # ''.join(list):将list转为str
    pred = " ".join(str(pred))
    #print (type(pred),pred)
    pred = re.findall(r"\d+",pred)
    pre_final.append(''.join(pred))
    #print("pre_final:",pre_final)

print ("pre_final:",pre_final)
#pre_final由字符串转换为0，1表示的数组。
for i in range(len(pre_final)):
    pre_final[i] = int(pre_final[i])

print ("labels:",labels)

print ("charCNN:")
"""print(accuracy_score(labels, pre_final),
    recall_score(labels, pre_final),
    precision_score(labels, pre_final),
    f1_score(labels, pre_final))"""


accuracy_final,recall_final,precision_final,f1_final = getMetrics(np.array(pre_final),np.array(labels))
print ("accuracy_final,recall_final,precision_final,f1_final:",accuracy_final,recall_final,precision_final,f1_final)

endTime = datetime.datetime.now()

#计算处理字符向量，构造模型，训练模型，验证模型，测试数据整个流程所用的时间

print ("endtime-begintime:",((endTime - beginTime).seconds * 1000 + (endTime - beginTime).microseconds / 1000),(endTime-beginTime).seconds)

print ("TrainendTime-TrainbeginTime:",((TrainendTime - TrainbeginTime).seconds * 1000 + (TrainendTime - TrainbeginTime).microseconds / 1000),(TrainendTime - TrainbeginTime).seconds)

print ("testTime:",((endTime - TestbeginTime).seconds * 1000 + (endTime - TestbeginTime).microseconds / 1000),(endTime - TestbeginTime).seconds)
