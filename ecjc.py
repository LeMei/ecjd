# -*- coding: utf-8 -*-
'''
@time: 2019/10/20 5:17 下午
@author: huguimin
@email: 718400742@qq.com
'''
'''
为了扩大cause的召回，将emotion 预测出来emotion clause的position作为特征传入cause的检测子任务中
'''

import numpy as np
import pickle as pk
import transformer as trans
import tensorflow as tf
import sys, os, time, codecs, pdb
import utils.tf_funcs as func
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
# embedding parameters ##
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
# input struct ##
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
tf.app.flags.DEFINE_integer('max_sen_len', 45, 'max number of tokens per sentence')
# model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 4, 'number of distinct class')
tf.app.flags.DEFINE_integer('emotion_n_class', 2, 'number of emotion distinct class')
tf.app.flags.DEFINE_integer('cause_n_class', 2, 'number of cause distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 15, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
tf.app.flags.DEFINE_string('save_dir', 'ECJL_ECPredictor', 'save dir')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('lr_assist', 0.001, 'learning rate of assist')
tf.app.flags.DEFINE_float('lr_main', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')
tf.app.flags.DEFINE_integer('run_times', 1, 'run times of this model')
tf.app.flags.DEFINE_integer('num_heads', 5, 'the num heads of attention')
tf.app.flags.DEFINE_integer('emotion_n_layers', 1, 'the layers of emotion transformer beside main')
tf.app.flags.DEFINE_integer('cause_n_layers', 1, 'the layers of cause transformer beside main')
tf.app.flags.DEFINE_integer('main_n_layers', 1, 'the layers of emotion-cause main transformer')
tf.app.flags.DEFINE_integer('assist_n_layers', 1, 'the layers of assist beside main')



def build_model(x, sen_len, doc_len, word_embedding, clause_position, embedding_pos, keep_prob1, keep_prob2, RNN=func.biLSTM):
    x = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    n_hidden = 2 * FLAGS.n_hidden

    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])
    with tf.name_scope('word_encode'):
        wordEncode = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'word_layer')
    wordEncode = tf.reshape(wordEncode, [-1, FLAGS.max_sen_len, n_hidden])

    with tf.name_scope('attention'):
        w1 = func.get_weight_varible('word_att_w1', [n_hidden, n_hidden])
        b1 = func.get_weight_varible('word_att_b1', [n_hidden])
        w2 = func.get_weight_varible('word_att_w2', [n_hidden, 1])
        senEncode = func.att_var(wordEncode, sen_len, w1, b1, w2)
        # (32*75,200)
    senEncode = tf.reshape(senEncode, [-1, FLAGS.max_doc_len, n_hidden]) #(32, 75, 200)

    n_feature = 2 * FLAGS.n_hidden
    out_units = 2 * FLAGS.n_hidden#200
    batch = tf.shape(senEncode)[0]#32
    pred_zeros = tf.zeros(([batch, FLAGS.max_doc_len, FLAGS.max_doc_len]))#(32,75,75)
    matrix = tf.reshape((1 - tf.eye(FLAGS.max_doc_len)), [1, FLAGS.max_doc_len, FLAGS.max_doc_len]) + pred_zeros  # 构造单位矩阵
    pred_emotion_assist_list, reg_emotion_assist_list, pred_emotion_assist_label_list = [], [], []
    pred_cause_assist_list, reg_cause_assist_list, pred_cause_assist_label_list = [], [], []




    if FLAGS.assist_n_layers > 1:

        '''******* emotion layer 1******'''
        emotion_senEncode = trans_func(senEncode, senEncode, n_feature, out_units, 'emotion_layer1')#(32,75,200)
        pred_emotion_assist, reg_emotion_assist = senEncode_emotion_softmax(emotion_senEncode, 'softmax_assist_w1', 'softmax_assist_b1', out_units, doc_len)
        #(32, 75,2)
        pred_emotion_assist_label = tf.cast(tf.reshape(tf.argmax(pred_emotion_assist, axis=2), [-1, 1, FLAGS.max_doc_len]), tf.float32)
        #(32, 75, 1)=>(32, 1, 75)

        pred_emotion_assist_position = tf.cast(tf.reshape(tf.argmax(pred_emotion_assist_label, axis=2), [-1, 1]) + 1, tf.int32) #emotion clause的所在位置，辅助clause的提取
        pred_clause_relative_position = tf.cast(tf.reshape(clause_position - pred_emotion_assist_position + 69, [-1, FLAGS.max_doc_len]), tf.float32) #基于emotion clause的相对位置 (32, 1, 75)
        pred_clause_relative_position *=func.getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len])
        pred_clause_relative_position = tf.cast(pred_clause_relative_position, tf.int32)
        pred_clause_rep_embed = tf.nn.embedding_lookup(embedding_pos, pred_clause_relative_position) #(32, 75, 50)

        pred_emotion_assist_label = (pred_emotion_assist_label + pred_zeros) * matrix  # 屏蔽预测为1的标签
        #matrix=>(32, 75, 75)
        #pred_assist_label=>(32, 75, 75)
        pred_emotion_assist_label_list.append(pred_emotion_assist_label)
        pred_emotion_assist_list.append(pred_emotion_assist)
        reg_emotion_assist_list.append(reg_emotion_assist)

        '''******* cause layer 1******'''
        cause_senEncode_assist = tf.concat([senEncode, pred_clause_rep_embed], axis=2)
        n_feature = out_units + FLAGS.embedding_dim_pos
        cause_senEncode = trans_func(cause_senEncode_assist, senEncode, n_feature, out_units, 'cause_layer')

        pred_cause_assist, reg_cause_assist = senEncode_cause_softmax(cause_senEncode, 'cause_softmax_assist_w1',
                                                                      'cause_softmax_assist_b1', out_units, doc_len)
        # (32, 75,2)
        pred_cause_assist_label = tf.cast(tf.reshape(tf.argmax(pred_cause_assist, axis=2), [-1, 1, FLAGS.max_doc_len]),
                                          tf.float32)
        # (32, 75, 1)=>(32, 1, 75)
        pred_cause_assist_label = (pred_cause_assist_label + pred_zeros) * matrix  # 屏蔽预测为1的标签
        # matrix=>(32, 75, 75)
        # pred_assist_label=>(32, 75, 75)
        pred_cause_assist_label_list.append(pred_cause_assist_label)
        pred_cause_assist_list.append(pred_cause_assist)
        reg_cause_assist_list.append(reg_cause_assist)

    for i in range(2, FLAGS.assist_n_layers):
        emotion_senEncode_assist = tf.concat([emotion_senEncode, pred_emotion_assist_label, pred_cause_assist_label], axis=2)  # (32, 75, 275)
        n_feature = out_units + 2 * FLAGS.max_doc_len  # 275
        emotion_senEncode = trans_func(emotion_senEncode_assist, emotion_senEncode, n_feature, out_units,
                                       'emotion_layer' + str(i))  # (32,75,200)

        pred_emotion_assist, reg_emotion_assist = senEncode_emotion_softmax(emotion_senEncode,
                                                                            'emotion_softmax_assist_w' + str(i),
                                                                            'emotion_softmax_assist_b' + str(i),
                                                                            out_units, doc_len)
        pred_emotion_assist_label = tf.cast(
            tf.reshape(tf.argmax(pred_emotion_assist, axis=2), [-1, 1, FLAGS.max_doc_len]), tf.float32)

        # pred_emotion_assist_position = tf.cast(tf.reshape(tf.argmax(pred_emotion_assist_label, axis=2), [-1, 1]),
        #                                        tf.float32) + 1  # emotion clause的所在位置，辅助clause的提取
        # pred_clause_relative_position = tf.reshape(clause_position - pred_emotion_assist_position,
        #                                            [-1, FLAGS.max_doc_len])  # 基于emotion clause的相对位置 (32, 1, 75)
        # pred_clause_relative_position *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len])
        # pred_clause_rep_embed = tf.nn.embedding_lookup(embedding_pos, pred_clause_relative_position)  # (32, 75, 50)

        pred_emotion_assist_label = (pred_emotion_assist_label + pred_zeros) * matrix
        pred_emotion_assist_label_list.append(pred_emotion_assist_label)

        pred_emotion_assist_label = tf.reduce_sum(pred_emotion_assist_label_list, axis=0)
        # 不同layer加和 pred_assist_label=>(32,75,75)

        pred_emotion_assist_list.append(pred_emotion_assist)
        reg_emotion_assist_list.append(reg_emotion_assist)

        cause_senEncode_assist = tf.concat([cause_senEncode, pred_cause_assist_label, pred_emotion_assist_label], axis=2)#(32, 75, 275)
        n_feature = out_units + 2 * FLAGS.max_doc_len #275
        cause_senEncode = trans_func(cause_senEncode_assist, cause_senEncode, n_feature, out_units, 'cause_layer' + str(i))#(32,75,200)

        pred_cause_assist, reg_cause_assist = senEncode_cause_softmax(cause_senEncode, 'cause_softmax_assist_w' + str(i), 'cause_softmax_assist_b' + str(i), out_units, doc_len)
        pred_cause_assist_label = tf.cast(tf.reshape(tf.argmax(pred_cause_assist, axis=2), [-1, 1, FLAGS.max_doc_len]), tf.float32)
        pred_cause_assist_label = (pred_cause_assist_label + pred_zeros) * matrix
        pred_cause_assist_label_list.append(pred_cause_assist_label)

        pred_cause_assist_label = tf.reduce_sum(pred_cause_assist_label_list, axis=0)
        #不同layer加和 pred_assist_label=>(32,75,75)

        pred_cause_assist_list.append(pred_cause_assist)
        reg_cause_assist_list.append(reg_cause_assist)



    '''*******Main******'''

    if FLAGS.main_n_layers > 1:
        senEncode_main = tf.concat([emotion_senEncode, cause_senEncode], axis=2)
        n_feature = 2 * out_units
        senEncode_main = trans_func(senEncode_main, senEncode_main, n_feature, out_units, 'main_layer1')
        senEncode_main = tf.concat([senEncode_main, pred_emotion_assist_label, pred_cause_assist_label], axis=2)
        n_feature = out_units + 2 * FLAGS.max_doc_len
        senEncode_main = trans_func(senEncode_main, senEncode_main, n_feature, out_units, 'main_layer2')
    else:
        senEncode_main = tf.concat([emotion_senEncode, cause_senEncode], axis=2)
        n_feature = 2 * out_units
        senEncode_main = trans_func(senEncode_main, senEncode_main, n_feature, out_units, 'main_layer1')
    pred, reg = senEncode_main_softmax(senEncode_main, 'softmax_w', 'softmax_b', out_units, doc_len)

    return pred, reg, pred_emotion_assist_list, reg_emotion_assist_list, pred_cause_assist_list, reg_cause_assist_list



def run():

    save_dir = 'result_data_{}/'.format(FLAGS.save_dir)
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    tf.reset_default_graph()
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)
    x_data, y_data, y_emotion_data, y_cause_data, sen_len_data, doc_len_data, word_embedding, doc_id_data, idx_word_dict, clause_position_data, embedding_pos, _ = func.load_data()

    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    embedding_pos = tf.constant(embedding_pos, dtype=tf.float32, name='embedding_pos')

    print('build model...')

    start_time = time.time()
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_emotion = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.emotion_n_class])
    y_cause = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.cause_n_class])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    clause_position = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)

    placeholders_emotion_assist = [x, y_emotion, sen_len, doc_len, clause_position, keep_prob1, keep_prob2]
    placeholders_cause_assist = [x, y_cause, sen_len, doc_len, clause_position, keep_prob1, keep_prob2]
    placeholders_main = [x, y, sen_len, doc_len, clause_position, keep_prob1, keep_prob2]

    pred, reg, pred_emotion_assist_list, reg_emotion_assist_list, pred_cause_assist_list, reg_cause_assist_list =\
        build_model(x, sen_len, doc_len, word_embedding,  clause_position, embedding_pos, keep_prob1, keep_prob2)

    with tf.name_scope('loss'):
        valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_op = - tf.reduce_sum(y * tf.log(pred)) / valid_num + reg * FLAGS.l2_reg
        loss_emotion_assist_list, loss_cause_assist_list = [], []
        for i in range(FLAGS.assist_n_layers - 1):
            loss_emotion_assist = - tf.reduce_sum(y_emotion * tf.log(pred_emotion_assist_list[i])) / valid_num + reg_emotion_assist_list[i] * FLAGS.l2_reg
            loss_emotion_assist_list.append(loss_emotion_assist)

            loss_cause_assist = - tf.reduce_sum(y_cause * tf.log(pred_cause_assist_list[i])) / valid_num + reg_cause_assist_list[i] * FLAGS.l2_reg
            loss_cause_assist_list.append(loss_cause_assist)


    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_op)
        optimizer_emotion_assist_list, optimizer_cause_assist_list = [], []
        for i in range(FLAGS.assist_n_layers - 1):
            if i == 0:
                optimizer_emotion_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_assist).minimize(loss_emotion_assist_list[i])
                optimizer_cause_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_assist).minimize(loss_cause_assist_list[i])

            else:
                optimizer_emotion_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_emotion_assist_list[i])
                optimizer_cause_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_assist).minimize(loss_cause_assist_list[i])

            optimizer_emotion_assist_list.append(optimizer_emotion_assist)
            optimizer_cause_assist_list.append(optimizer_cause_assist)




    true_y_op = tf.argmax(y, 2)
    pred_y_op = tf.argmax(pred, 2)
    emotion_true_y_op = tf.argmax(y_emotion, 2)
    cause_true_y_op = tf.argmax(y_cause, 2)
    pred_y_emotion_assist_op_list, pred_y_cause_assist_op_list = [], []
    for i in range(FLAGS.assist_n_layers - 1):
        pred_y_emotion_assist_op = tf.argmax(pred_emotion_assist_list[i], 2)
        pred_y_emotion_assist_op_list.append(pred_y_emotion_assist_op)

        pred_y_cause_assist_op = tf.argmax(pred_cause_assist_list[i], 2)
        pred_y_cause_assist_op_list.append(pred_y_cause_assist_op)

    print('build model done!\n')

    prob_list_pr, y_label = [], []
    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        kf, fold, SID = KFold(n_splits=10), 1, 0
        Id = []
        p_list, r_list, f1_list = [], [], []
        for train, test in kf.split(x_data):
            tr_x, tr_y, tr_y_emotion, tr_y_cause, tr_sen_len, tr_doc_len, tr_doc_id, tr_clause_pos = map(lambda x: x[train],
                                                                  [x_data, y_data, y_emotion_data, y_cause_data,
                                                                   sen_len_data, doc_len_data, doc_id_data, clause_position_data])
            te_x, te_y, te_y_emotion, te_y_cause, te_sen_len, te_doc_len, te_doc_id, te_clause_pos = map(lambda x: x[test],
                                                                  [x_data, y_data, y_emotion_data, y_cause_data,
                                                                   sen_len_data, doc_len_data, doc_id_data, clause_position_data])

            precision_list, recall_list, FF1_list = [], [], []
            pre_list, true_list, pre_list_prob = [], [], []
            # file_dir = './ECPredictor_demo_assistlayer4_mainlayer1/fold{}/'.format(fold)
            # if not os.path.exists(file_dir):
            #     os.mkdir(file_dir)
            sess.run(tf.global_variables_initializer())
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1 = 0.0
            emo_max_f1 = 0.0
            cause_max_f1 = 0.0
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))


            for layer in range(FLAGS.assist_n_layers - 1):
                if layer == 0:
                    training_iter = FLAGS.training_iter
                else:
                    training_iter = FLAGS.training_iter - 5
                for i in range(training_iter):
                    '''预训练情感分类'''
                    emotion_step = 1
                    for train, _ in get_emotion_batch_data(tr_x, tr_y_emotion, tr_sen_len, tr_doc_len, tr_clause_pos, FLAGS.keep_prob1,
                                                   FLAGS.keep_prob2, FLAGS.batch_size):
                        _, loss, pred_y, true_y, doc_len_batch = sess.run(
                            [optimizer_emotion_assist_list[layer], loss_emotion_assist_list[layer], pred_y_emotion_assist_op_list[layer],
                             emotion_true_y_op, doc_len],
                            feed_dict=dict(zip(placeholders_emotion_assist, train)))
                        acc_assist, p_assist, r_assist, f1_assist = func.acc_prf_binary(pred_y, true_y, doc_len_batch)
                        if emotion_step % 10 == 0:
                            print('Emotion {}: epoch {}: step {}: loss {:.4f} acc {:.4f} p {:.4f}'.format(layer + 1, i + 1,
                                                                                                    emotion_step, loss,
                                                                                                    acc_assist,
                                                                                                    p_assist))
                        emotion_step = emotion_step + 1
                    """****test emotion extraction****"""
                    test = [te_x, te_y_emotion, te_sen_len, te_doc_len, te_clause_pos, 1., 1.]
                    loss, pred_y, true_y, doc_len_batch = sess.run([loss_emotion_assist_list[layer], pred_y_emotion_assist_op_list[layer], emotion_true_y_op, doc_len],
                                                                   feed_dict=dict(zip(placeholders_emotion_assist, test)))
                    acc_test, p_test, r_test, f1_test = func.acc_prf_binary(pred_y, true_y, doc_len_batch)
                    if emo_max_f1 < f1_test:
                        emo_max_f1 = f1_test
                    print('\nemotion-test: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                        i + 1, loss, acc_test, p_test, r_test, f1_test, emo_max_f1))
                    # file_name = file_dir + 'emotion_test_layer{}_epoch{}.txt'.format(layer + 1, i+1)
                    # func.output_pred(file_name, te_doc_id, te_x, te_doc_len, te_sen_len, true_y, pred_y, idx_word_dict)


                    """预训练原因分类"""
                    cause_step = 1
                    for train, _ in get_cause_batch_data(tr_x, tr_y_cause, tr_sen_len, tr_doc_len, tr_clause_pos,
                                                         FLAGS.keep_prob1,
                                                         FLAGS.keep_prob2, FLAGS.batch_size):
                        _, loss, pred_y, true_y, doc_len_batch = sess.run(
                            [optimizer_cause_assist_list[layer], loss_cause_assist_list[layer],
                             pred_y_cause_assist_op_list[layer],
                             cause_true_y_op, doc_len],
                            feed_dict=dict(zip(placeholders_cause_assist, train)))
                        acc_assist, p_assist, r_assist, f1_assist = func.acc_prf_binary(pred_y, true_y, doc_len_batch)
                        if cause_step % 10 == 0:
                            print(
                                'Cause {}: epoch {}: step {}: loss {:.4f} acc {:.4f} p {:.4f}'.format(layer + 1, i + 1,
                                                                                                      cause_step, loss,
                                                                                                      acc_assist,
                                                                                                      p_assist))
                        cause_step = cause_step + 1
                    """****test cause extraction****"""
                    test = [te_x, te_y_cause, te_sen_len, te_doc_len, te_clause_pos, 1., 1.]
                    loss, pred_y, true_y, doc_len_batch = sess.run([loss_cause_assist_list[layer],
                                                                   pred_y_cause_assist_op_list[layer],
                                                                   cause_true_y_op, doc_len], feed_dict=dict(zip(placeholders_cause_assist, test)))
                    acc_test, p_test, r_test, f1_test = func.acc_prf_binary(pred_y, true_y, doc_len_batch)
                    if cause_max_f1 < f1_test:
                        cause_max_f1 = f1_test
                    print(
                    '\ncause-test: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                        i + 1, loss, acc_test, p_test, r_test, f1_test, cause_max_f1))

                    # file_name = file_dir + 'cause_test_layer{}_epoch{}.txt'.format(layer+1, i+1)
                    # func.output_pred(file_name, te_doc_id, te_x, te_doc_len, te_sen_len, true_y, pred_y, idx_word_dict)

            '''*********Train********'''
            for epoch in range(FLAGS.training_iter):
                step = 1
                for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, tr_clause_pos, FLAGS.keep_prob1,
                                               FLAGS.keep_prob2, FLAGS.batch_size):
                    _, loss, pred_y, true_y, pred_prob, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_op, true_y_op, pred, doc_len],
                        feed_dict=dict(zip(placeholders_main, train)))
                    acc, p, r, f1 = func.acc_prf_multiclass(pred_y, true_y, doc_len_batch)
                    if step % 5 == 0:
                        print('epoch {}: step {}: loss {:.4f} acc {:.4f} p {:.4}'.format(epoch + 1, step, loss, acc, p))
                    step = step + 1

                '''*********Test********'''
                test = [te_x, te_y, te_sen_len, te_doc_len, te_clause_pos, 1., 1.]
                loss, pred_y, true_y, pred_prob = sess.run(
                    [loss_op, pred_y_op, true_y_op, pred], feed_dict=dict(zip(placeholders_main, test)))

                end_time = time.time()

                true_list.append(true_y)
                pre_list.append(pred_y)
                pre_list_prob.append(pred_prob)

                acc, p, r, f1 = func.acc_prf_multiclass(pred_y, true_y, te_doc_len)
                precision_list.append(p)
                recall_list.append(r)
                FF1_list.append(f1)
                if f1 > max_f1:
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                print('\ntest: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                    epoch + 1, loss, acc, p, r, f1, max_f1))
                # file_name = file_dir + 'emotion_cause_test_{}.txt'.format(epoch+1)
                # func.output_pred(file_name, te_doc_id, te_x, te_doc_len, te_sen_len, true_y, pred_y, idx_word_dict)

            Id.append(len(te_x))
            SID = np.sum(Id) - len(te_x)
            _, maxIndex = func.maxS(FF1_list)
            print("maxIndex:", maxIndex)
            print('Optimization Finished!\n')
            pred_prob = pre_list_prob[maxIndex]

            for i in range(pred_y.shape[0]):
                for j in range(te_doc_len[i]):
                    prob_list_pr.append(pred_prob[i][j][1])
                    y_label.append(true_y[i][j])

            print("*********prob_list_pr", len(prob_list_pr))
            print("*********y_label", len(y_label))

            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)

        print("running time: ", str((end_time - start_time) / 60.))
        print_training_info()
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])

        print("f1_score in 10 fold: {}\naverage : {} {} {}\n".format(np.array(f1_list).reshape(-1, 1), round(p, 4),
                                                                     round(r, 4), round(f1, 4)))
        return p, r, f1


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, learning_rate-{}, keep_prob1-{}, num_heads-{}, assist_n_layers-{}, main_n_layers-{}'.format(
        FLAGS.batch_size, FLAGS.lr_main, FLAGS.keep_prob1, FLAGS.num_heads, FLAGS.assist_n_layers, FLAGS.main_n_layers))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, y, sen_len, doc_len, clause_pos, keep_prob1, keep_prob2, batch_size, test=False):
    for index in func.batch_index(len(y), batch_size, test):
        feed_list = [x[index], y[index], sen_len[index], doc_len[index], clause_pos[index], keep_prob1, keep_prob2]
        yield feed_list, len(index)

def get_emotion_batch_data(x,  y_emotion, sen_len, doc_len, clause_pos, keep_prob1, keep_prob2, batch_size, test=False):
    for index in func.batch_index(len(y_emotion), batch_size, test):
        feed_list = [x[index], y_emotion[index], sen_len[index], doc_len[index], clause_pos[index], keep_prob1, keep_prob2]
        yield feed_list, len(index)

def get_cause_batch_data(x,  y_cause, sen_len, doc_len, clause_pos, keep_prob1, keep_prob2, batch_size, test=False):
    for index in func.batch_index(len(y_cause), batch_size, test):
        feed_list = [x[index], y_cause[index], sen_len[index], doc_len[index], clause_pos[index], keep_prob1, keep_prob2]
        yield feed_list, len(index)



def senEncode_main_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=FLAGS.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, FLAGS.n_class])
    b = func.get_weight_varible(b_varible, [FLAGS.n_class])
    pred = tf.matmul(s, w) + b
    pred *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, FLAGS.max_doc_len, FLAGS.n_class])
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg

def senEncode_emotion_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=FLAGS.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, FLAGS.emotion_n_class])
    b = func.get_weight_varible(b_varible, [FLAGS.emotion_n_class])
    pred = tf.matmul(s, w) + b
    pred *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, FLAGS.max_doc_len, FLAGS.emotion_n_class])
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg

def senEncode_cause_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=FLAGS.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, FLAGS.cause_n_class])
    b = func.get_weight_varible(b_varible, [FLAGS.cause_n_class])
    pred = tf.matmul(s, w) + b
    pred *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, FLAGS.max_doc_len, FLAGS.cause_n_class])
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg

def trans_func(senEncode_dis, senEncode, n_feature, out_units, scope_var):
    senEncode_assist = trans.multihead_attention(queries=senEncode_dis,
                                            keys=senEncode_dis,
                                            values=senEncode,
                                            units_query=n_feature,
                                            num_heads=FLAGS.num_heads,
                                            dropout_rate=0,
                                            is_training=True,
                                            scope=scope_var)
    senEncode_assist = trans.feedforward_1(senEncode_assist, n_feature, out_units)
    return senEncode_assist

def main(_):
    grid_search = {}
    params = {"assist_n_layers": [3], "main_n_layers":[2]}


    params_search = list(ParameterGrid(params))

    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(FLAGS, key, value)
        p_list, r_list, f1_list = [], [], []
        for i in range(FLAGS.run_times):
            print("*************run(){}*************".format(i + 1))
            p, r, f1 = run()
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

        for i in range(FLAGS.run_times):
            print(round(p_list[i], 4), round(r_list[i], 4), round(f1_list[i], 4))
        print("avg_prf: ", np.mean(p_list), np.mean(r_list), np.mean(f1_list))

        grid_search[str(param)] = {"PRF": [round(np.mean(p_list), 4), round(np.mean(r_list), 4), round(np.mean(f1_list), 4)]}

    for key, value in grid_search.items():
        print("Main: ", key, value)



if __name__ == '__main__':
    tf.app.run()



