# -*- encoding:utf-8 -*-
'''
@time: 2019/10/15
@author: huguimin
@email: 718400742@qq.com
'''
import codecs
import re
import numpy as np
import sys
import jieba as jb

reload(sys)

sys.setdefaultencoding('utf8')

def process(input_file, output_file):
    input_lines = codecs.open(input_file, 'r', 'utf-8')
    emotion_pattern = re.compile(r'f](.+)\[/f')
    cause_pattern = re.compile(r'\d.*](.+)\[/.*\d')
    rep_emotion_pattern = re.compile(r'[f\][/]')
    rep_cause_pattern = re.compile(r'[([\-{v,m}\d/*\])]')
    lines = []
    emotion_id = []
    causes_id = []
    for doc_id, line in enumerate(input_lines):
        line = line.replace(u'“',"").strip()
        line = line.replace(u"”","").strip()
        line = line.strip().split(',')
        raw_text = line[-1].strip()
        clause_cause_id = []
        doc_line = []
        sentences = raw_text.split('。')
        clauses = [sen.strip().split(u'，') for sen in sentences]
        emotion_word = 'none'
        global_clause_id = 0
        for sen_id, sen in enumerate(clauses):
            for clause_id, clause in enumerate(sen):
                if not clause: continue
                global_clause_id = global_clause_id + 1
                clause = clause.strip()
                emotion = re.findall(emotion_pattern,clause)
                cause = re.findall(cause_pattern,clause)
                emotion_lable = 0
                cause_lable = 0
                if len(emotion):
                    emotion_lable = 1
                    emotion_word = emotion[0]
                    emotion_id.append(global_clause_id)
                    clause = re.subn(rep_emotion_pattern, '', clause)[0]
                if len(cause):
                    cause_lable = 1
                    clause_cause_id.append(global_clause_id)
                    clause = re.subn(rep_cause_pattern, '', clause)[0]

                new_line = [doc_id+1, sen_id+1, global_clause_id, clause_id+1, emotion_word, emotion_lable, cause_lable, clause]
                doc_line.append(new_line)
        lines.append(doc_line)
        causes_id.append(clause_cause_id)

    f = open(output_file, 'a+')
    for doc, clauses in enumerate(lines):
        emotion_pos = emotion_id[doc]
        print(doc)
        emotion_word = clauses[emotion_pos-1][4]
        clauses = np.array(clauses)
        cause_pos = np.array(causes_id[doc]) -1
        for cla_id, cla in enumerate(clauses):
            rpe = cla_id + 1 - emotion_pos
            rpc = '|'.join(map(str, list(cla_id - cause_pos)))
            words = ' '.join(jb.cut(cla[7]))
            new_line = map(str, [cla[0], cla[1], cla[2], cla[3], rpe, rpc, emotion_word, cla[5], cla[6], words])
            # new_line = map(str, [cla[0], cla[1], cla[2], cla[3], rpe, rpc, emotion_word, cla[5], cla[6], cla[7]])
            # print(new_line)
            f.write(','.join(new_line))
            f.write('\n')



    f.close()

def do_statics(input_file):
    input_lines = codecs.open(input_file, 'r', 'utf-8')
    emotion_cause = 0
    for line in input_lines:
        line = line.strip().split(',')
        emotion_tag, cause_tag = int(line[7]), int(line[8])
        if emotion_tag == 1 and emotion_tag == cause_tag:
            emotion_cause = emotion_cause + 1

    print('the number of emotion same as cause {}'.format(emotion_cause))


input_file = '../data/datacsv_2105.csv'
output_file = "../data/data.csv"
process(input_file, output_file)

# input_file = '../data/data.csv'
# do_statics(input_file)





