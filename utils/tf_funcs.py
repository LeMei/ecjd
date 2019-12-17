# -*- encoding:utf-8 -*-
'''
@time: 2019/10/15
@author: huguimin
@email: 718400742@qq.com
'''

import numpy as np
import tensorflow as tf
import os
import json
import random
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
path = './data/'
max_doc_len = 75
max_sen_len = 45
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def load_data():
    x = pk.load(open(path + 'x.txt', 'rb'))
    y = pk.load(open(path + 'y.txt', 'rb'))
    y_emotion = pk.load(open(path + 'y_emotion.txt', 'rb'))
    y_cause = pk.load(open(path + 'y_cause.txt', 'rb'))
    sen_len = pk.load(open(path + 'sen_len.txt', 'rb'))
    doc_len = pk.load(open(path + 'doc_len.txt', 'rb'))
    # relative_pos = pk.load(open(path + 'relative_pos.txt', 'rb'))
    embedding = pk.load(open(path + 'embedding.txt', 'rb'))
    embedding_pos = pk.load(open(path + 'embedding_pos.txt', 'rb'))
    doc_id = pk.load(open(path + 'doc_id.txt', 'rb'))
    idx_word = pk.load(open(path + 'idx_word.txt', 'rb'))
    idx_word_dict = json.loads(idx_word)
    clause_position = pk.load(open(path + 'clause_position.txt', 'rb'))
    time_position = pk.load(open(path + 'time_position.txt', 'rb'))
    print('x.shape {} \ny.shape {} \ny_emotion.shape {} \ny_cause.shape{} \nsen_len.shape {} \ndoc_len.shape {}\ndoc_id.shape {}\nclause_position.shape {}\ntime_position.shape {}'
          .format(x.shape, y.shape, y_emotion.shape, y_cause.shape, sen_len.shape, doc_len.shape, doc_id.shape, clause_position.shape, time_position.shape))
    return x, y, y_emotion, y_cause, sen_len, doc_len, embedding, doc_id, idx_word_dict, clause_position, embedding_pos, time_position


def acc_prf_binary(pred_y, true_y, doc_len):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return acc, p, r, f1

def acc_prf_multiclass(pred_y, true_y, doc_len):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, p, r, f1

def create_confusion_matrix(pred_y, true_y, doc_len):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    matrix = confusion_matrix(y_true, y_pred)
    return matrix



def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if not test:
        random.shuffle(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size: (i + 1) * batch_size]
        if not test and len(ret) < batch_size:
            break
        yield ret


def get_weight_varible(name, shape):
    return tf.get_variable(name, initializer=tf.random_uniform(shape, -0.01, 0.01))


def getmask(length, max_len, out_shape):
    '''
    length shape:[batch_size]
    '''
    # 转换成 0 1
    ret = tf.cast(tf.sequence_mask(length, max_len), tf.float32)
    return tf.reshape(ret, out_shape)


def biLSTM(inputs, length, n_hidden, scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    return tf.concat(outputs, 2)


def softmax_by_length(inputs, length):
    '''
    input shape:[batch_size, 1, max_len]
    (batch_size*max_doc_len, 1, max_len)
    length shape:[batch_size]
    return shape:[batch_size, 1, max_len]
    (batch_size*max_doc_len, 1, max_len)
    '''
    inputs = tf.exp(tf.cast(inputs, tf.float32))
    inputs *= getmask(length, tf.shape(inputs)[2], tf.shape(inputs))
    _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
    return inputs / _sum


def att_var(inputs, length, w1, b1, w2):
    '''
    input shape:[batch_size*max_doc_len, max_sen_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len, n_hidden = (tf.shape(inputs)[1], tf.shape(inputs)[2])  # (45, n_hidden)
    # (batch_size*max_doc_len*max_sen_len, n_hidden)
    tmp = tf.reshape(inputs, [-1, n_hidden])
    u = tf.tanh(tf.matmul(tmp, w1) + b1)
    alpha = tf.reshape(tf.matmul(u, w2), [-1, 1, max_len])
    #(batch_size*max_doc_len, 1, max_len)
    alpha = softmax_by_length(alpha, length)
    #(batch_size * max_doc_len, 1, max_len)
    return tf.reshape(tf.matmul(alpha, inputs), [-1, n_hidden])
    #tf.matmul(alpha, inputs)=>(batch_size*max_doc_len, 1, n_hidden)
    #(batch_size*max_doc_len,n_hidden)

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

def memory_networks(query, Ain, Bin, C, lindim, edim, mem_size, batch_size, nhop=3):
    '''
    query : (32, 75, 200)
    A_in : (32, 75, 45, 250)
    B_in : (32, 75, 45, 250)
    C : [n_hidden, n_hidden]

    '''
    memory_list = []
    max_len, n_hidden = (tf.shape(query)[1], tf.shape(query)[2])
    memory_list.append(query)
    for h in xrange(nhop):
        hid3dim = tf.reshape(memory_list[-1], [-1, max_len, 1, edim])  # (32, 75, 1,200)
        Aout = tf.matmul(hid3dim, Ain,
                         adjoint_b=True)  # (32, 75, 1, 200) * (32, 75, 45, 200) => (32, 75, 1, 45)
        Aout2dim = tf.reshape(Aout, [-1, mem_size])  # (32*75*1, 45)
        P = tf.nn.softmax(Aout2dim) #(32*75, 45)

        probs3dim = tf.reshape(P, [-1, max_len, 1, mem_size]) #(32, 75, 1, 45)
        Bout = tf.matmul(probs3dim, Bin)  # (32, 75, 1, 45)*(32, 75, 45, 200)=>(32,75,1, 200)
        Bout2dim = tf.reshape(Bout, [-1, edim])  # (32*75, 200)

        Cout = tf.matmul(tf.reshape(memory_list[-1],[-1, edim]), C)  # # (32*75, 200)*(200,200)=># (32*75, 200)
        Dout = tf.reshape(tf.add(Cout, Bout2dim), [-1, max_len, edim])  # # (32*75, 200)


        if lindim == edim:
            memory_list.append(Dout)
        elif lindim == 0:
            memory_list.append(tf.nn.relu(Dout))
        else:
            F = tf.slice(Dout, [0, 0, 0], [batch_size, max_len, lindim])  # (32, 75) 前75维
            G = tf.slice(Dout, [0, 0, lindim], [batch_size, max_len, edim - lindim])  # (32, 75)后75维
            K = tf.nn.relu(G)
            memory_list.append(tf.concat(axis=2, values=[F, K]))

    return memory_list[-1]

def multi_memory_networks(query, context, embedding, edim, nhop=3):
    """"
    query : [32, 75, 200]
    pos_encoding : [75, 200]
    context:[32, 75, 45]
    """
    C_list = []
    pos_encoding = tf.constant(position_encoding(max_sen_len, edim), name='pos_encoding') #(45,200)
    for hopn in range(nhop):
        with tf.variable_scope('hop_{}'.format(hopn)):
            C_list.append(tf.Variable(embedding, name="C"))
    u_0 = query #(-1,75,45,200)*(45,200) => (-1, 75, 200)
    u = [u_0]

    for hopn in range(nhop):
        if hopn == 0:
            m_emb_A = tf.nn.embedding_lookup(embedding, context) #(-1, 75, 45, 200)
            m_A = m_emb_A * pos_encoding  # (32, 75, 45, 200) * (45, 200) => (32, 75, 45, 200)

        else:
            with tf.variable_scope('hop_{}'.format(hopn - 1)):
                m_emb_A = tf.nn.embedding_lookup(C_list[hopn - 1], context)
                m_A = m_emb_A * pos_encoding  # (32, 75, 45，200)

        # hack to get around no reduce_dot
        u_temp = tf.reshape(u[-1], [-1, max_doc_len, edim, 1])  # (32, 75, 1, 200)
        dotted = tf.reshape(tf.matmul(m_A, u_temp), [-1, max_sen_len])  # (32, 75, 45, 200) * (32, 75, 200, 1)=>((32, 75, 1, 200)=>(32, 75, 45, 1 )
        # Calculate probabilities
        probs = tf.reshape(tf.nn.softmax(dotted), [-1, max_doc_len, max_sen_len]) #(-1, 75, 45)

        probs_temp = tf.transpose(tf.expand_dims(probs, -1) [0, 2, 1])  # (-1, 1, 45)
        with tf.variable_scope('hop_{}'.format(hopn)):
            m_emb_C = tf.nn.embedding_lookup(C_list[hopn], context)  # (-1, 75, 45, 200)
        m_C = tf.reduce_sum(m_emb_C * pos_encoding, 2)  # (32, 75, 45, 200)

        # c_temp = tf.transpose(m_C, [0, 2, 1])  # (-1, 40, 50)
        o_k = m_C * probs_temp  # (32, 75, 45, 200)*(32, 75, 45, 1)=>(-1, 75，200)


        u_k = u[-1] + o_k  # (-1, 75, 200)

        u.append(u_k)

    return u[-1]









def maxS(alist):
    maxScore = 0.0
    maxIndex = -1
    for i in range(len(alist)):
        if alist[i] > maxScore:
            maxScore = alist[i]
            maxIndex = i
    return maxScore, maxIndex


def fun1(prob_pred, doc_len):
    ret = []
    for i in range(len(prob_pred)):
        ret.extend(list(prob_pred[i][:doc_len[i]]))
    return np.array(ret)


def output_pred(file_name, doc_id, x, doc_len, sen_len, true, pred, word_idx_rev):
    g = open(file_name, 'w')
    for i in range(len(doc_id)):
        label = true[i].argmax()+1
        g.write(str(doc_id[i]) + ' ' + str(doc_len[i]) + ' ' + str(label) + '\n')
        for j in range(doc_len[i]):
            clause = ''
            for k in range(sen_len[i][j]):
                key = x[i][j][k].astype(np.int64)
                clause = clause + word_idx_rev[str(key)] + ' '
            g.write(
                str(j + 1) + ', ' + str(pred[i][j]) + ', ' + clause + '\n')
    print 'write {} done'.format(file_name)

