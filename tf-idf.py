# encoding: utf-8
import os
import math

FILE_PATH='./ICML/'
STOP_WORD_PATH='./stop_words.txt'


def get_stopwords():
    stop_words=[]
    for line in open(STOP_WORD_PATH):
        line=line.strip()
        stop_words.append(line)
    return stop_words

def get_doc(stopwords):
    all_doc_list = []
    all_doc_words = []
    doc_count=0
    for filename in os.listdir(FILE_PATH):
        if find_seventh_file(filename):
            print '第七个文件的长度:',len(os.listdir(FILE_PATH+filename))
            seventh_begin_index=doc_count
            seventh_last_index=seventh_begin_index+len(os.listdir(FILE_PATH + filename))
        for doc in os.listdir(FILE_PATH + filename):
            doc_count+=1
            f = open(FILE_PATH + filename + '/' + doc)
            doc_mat = []
            for line in f.readlines():
                currentline = line.strip()
                if len(currentline) != 0:
                    doc_mat.extend([w for w in currentline.split(' ') if w not in stopwords])
            all_doc_list.append(doc_mat)
            all_doc_words.extend(doc_mat)
    all_doc_words=set(all_doc_words)
    return all_doc_list,all_doc_words,seventh_begin_index,seventh_last_index

def tf(doc):
    tf_map = {}
    for term in doc:
        if term not in tf_map.keys():
            tf_map[term] = 1
        else:
            tf_map[term] = tf_map[term] + 1
    sum_terms=sum(tf_map.values())
    for term in tf_map.keys():
        tf_map[term]=float(tf_map[term])/sum_terms
    return tf_map


def idf(all_doc_list,all_doc_words):
    idf_map={}
    tf_map_list=[]
    for doc in all_doc_list:
        tf_map=tf(doc)
        tf_map_list.append(tf_map)
    for term in all_doc_words:
        idf_map[term]=0
        for map in tf_map_list:
            if term in map.keys():
                idf_map[term]=idf_map[term]+1
    for term in idf_map.keys():
        idf_map[term]=math.log(float(len(all_doc_list))/(1+idf_map[term]))
    return idf_map


def get_doc_wordvector(all_doc_list,all_doc_words):
    idf_map=idf(all_doc_list,all_doc_words)
    all_doc_wordvector=[]
    for doc in all_doc_list:
        doc_wordvector = [0] * len(all_doc_words)
        tf_map=tf(doc)
        for term in tf_map.keys():
            tf_idf=tf_map[term]*idf_map[term]
            doc_wordvector[list(all_doc_words).index(term)]=tf_idf
        all_doc_wordvector.append(doc_wordvector)
    return all_doc_wordvector

def construct(line):
    new_line=[]
    for i,iterm in enumerate(line):
        if iterm==0:
            continue
        new_iterm="%s:%s"%(i+1,iterm)
        new_line.append(new_iterm)
    new_line=",".join(new_line)
    new_line += "\n"
    return new_line

def find_seventh_file(filename):
    if filename.split('.')[0]=='7':
        return True
    else:
        return False


if __name__=='__main__':
    print '开始读取文件...'
    all_doc_list, all_doc_words,seventh_begin_index,seventh_last_index= get_doc(get_stopwords())
    print "结果为",seventh_begin_index,seventh_last_index
    print '读取文件成功...'
    print '文档数目%s，文档单词%s'%(all_doc_list.__len__(),all_doc_words.__len__())
    doc_vector=get_doc_wordvector(all_doc_list,all_doc_words)
    print '开始写入文件...'
    f = open('./result', 'w')
    for i in range(seventh_begin_index,seventh_last_index):
        construct_line=construct(doc_vector[i])
        f.write(construct_line)
    f.close()
    print '写入文件成功...'
