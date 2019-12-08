import os
import collections
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DataSet(object):
    def __init__(self, file):
        self.file = file
        self.data_index = 0
        self._build_dataset()
    def _build_dataset(self):
        if not os.path.exists(self.file):
            raise ValueError("File does not exist --> %s" % self.file)
        f = open(self.file, mode='rt', encoding='utf8')
        self.data = tf.compat.as_str(f.read()).split()
        print("the length of data %d" % len(self.data))
        print("data value:" , self.data)
        if f:
            f.close()
        c = collections.Counter(self.data).most_common()
        print(c)
        self.vocab_size = len(c)
        self.counter = c.insert(0, ('UNK', -1))
        self.vocab_size += 1
        self.word2id = dict()
        self.id2word = dict()
        for word, _ in c:
            self.word2id[word] = len(self.word2id)#建立word到id的映射
            self.id2word[len(self.id2word)] = word#建立id到word的映射
    def gen_batch_inputs(self, batch_size, skip_window):
        raise NotImplementedError()


class SkipGramDataSet(DataSet):
    def gen_batch_inputs(self, batch_size, window_size):
        features = np.ndarray(shape=(batch_size,), dtype=np.int32)#shape(4,)和shape(4,1)是维度不一样
        labels = np.ndarray(shape=(batch_size,), dtype=np.int32)
        i = 0
        while True:
            if self.data_index == len(self.data):
                self.data_index = 0
            left = max(0, self.data_index - window_size)
            right = min(len(self.data), self.data_index + window_size + 1)
            for k in range(left, right):
                if k != self.data_index:
                    features[i] = self.word2id[self.data[self.data_index]]
                    labels[i] = self.word2id[self.data[k]]
                    i += 1
                    if i == batch_size:
                        return features, labels
            self.data_index += 1


dataset = SkipGramDataSet('/Users/zhongqiqiang/nease_projrct/recommend_recall/word2vec/test.txt')
VOCAB_SIZE = dataset.vocab_size
EMBEDDING_SIZE = 128
SKIP_WINDOW = 2
NUM_SAMPLED = 64
BATCH_SIZE = 32
WINDOW_SIZE = 2
LOG_DIR = "./tmp/word2vec"
TRAIN_STEPS = 10000
LEARNING_RATE = 0.1

class Word2Vec(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("inputs"):
                self.x = tf.placeholder(shape=(None, VOCAB_SIZE), dtype=tf.float32)
                self.y = tf.placeholder(shape=(None, VOCAB_SIZE), dtype=tf.float32)
            with tf.name_scope("layer1"):
                self.W1 = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1, 1),dtype=tf.float32)
                self.b1 = tf.Variable(tf.random_normal([EMBEDDING_SIZE]),dtype=tf.float32)
            hidden = tf.add(self.b1, tf.matmul(self.x, self.W1))
            with tf.name_scope("layer2"):
                self.W2 = tf.Variable(tf.random_uniform([EMBEDDING_SIZE, VOCAB_SIZE], -1, 1),dtype=tf.float32)
                self.b2 = tf.Variable(tf.random_normal([VOCAB_SIZE]),dtype=tf.float32)
            self.prediction = tf.nn.softmax(tf.add(tf.matmul(hidden, self.W2), self.b2))
            """
            损失函数
            """
            log = self.y * tf.log(self.prediction)
            self.loss = tf.reduce_mean(-tf.reduce_sum(log, reduction_indices=[1], keepdims=True))
            self.opt = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)

    def _one_hot_input(self, dataset):
        features, labels = dataset.gen_batch_inputs(BATCH_SIZE, WINDOW_SIZE)
        f, l = [], []
        for w in features:
            tmp = np.zeros([VOCAB_SIZE])
            tmp[w] = 1
            f.append(tmp)
        for w in labels:
            tmp = np.zeros(VOCAB_SIZE)
            tmp[w] = 1
            l.append(tmp)
        return f, l

    def train(self, dataset, n_iters, ):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_iters):
                features, labels = self._one_hot_input(dataset)

                predi, loss = sess.run([self.prediction, self.loss],
                                       feed_dict={
                                           self.x: features,
                                           self.y: labels
                                       })
                print("loss:%s" % loss)

    def predict(self):
        pass

    def nearest(self, n):
        pass

    def similarity(self, a, b):
        pass


word2vec = Word2Vec()
word2vec.train(dataset, TRAIN_STEPS)
