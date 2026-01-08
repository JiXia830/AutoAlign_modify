# coding: utf-8

from rdflib import Graph
import random
import numpy as np
import tensorflow as tf
import math
import datetime as dt
import pickle as cPickle
from functools import reduce
import time
from rdflib import URIRef
import os
DEVICE = "0"
os.environ["CUDA_VISIBLE_DEVICES"]=DEVICE

from kitchen.text.converters import getwriter, to_bytes, to_unicode
from kitchen.i18n import get_translation_object
translations = get_translation_object('example')
_ = translations.ugettext
b_ = translations.lgettext
### Combine two KG
dataset_name = 'example'
path = 'MS/'
lgd_filename = './data/'+path+'model_data'+'.ttl'
dbp_filename = './data/'+path+'book_data'+'.ttl'
predicate_graph_name = 'D:\AutoAlign\data\MS\examplebook_pred_prox_graph_matched.pickle'
predicate_graph = cPickle.load(open(predicate_graph_name, 'rb'))
# map_file = '../data/'+path+'mapping_'+dataset_name+'.ttl'
map_file = './data/'+path+'map'+'.ttl'
aggregation = 'W'

graph = Graph()
graph.parse(location=lgd_filename, format='nt')
graph.parse(location=dbp_filename, format='nt')

map_graph = Graph()
map_graph.parse(location=map_file, format='nt')


def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

entity_label_dict = dict()

for s,p,o in graph:
    if str(p) in ['http://example.org/has_label','http://book.org/label']:
        entity_label_dict[s] = str(o)


num_subj_triple = dict()
for s,p,o in graph:
    if num_subj_triple.get(s) == None:
        num_subj_triple[s] = 1
    else:
        num_subj_triple[s] += 1


# YAGO
intersection_predicates = ['http://example.org/has_name','http://example.org/has_label',\
                           'http://example.org/has_value','ttp://book.org/label',\
                            'http://example.org/describle','http://book.org/context',\
                            'http://book.org/purpose','http://book.org/purpose','http://book.org/label_context','http://book.org/constrainted']
intersection_predicates_uri = ['http://example.org/Part_Structure','http://example.org/Structural_Self-Analysis',\
                               'http://example.org/Structural_Function','http://example.org/Structural_Constraint',\
                                'http://example.org/Structural_Instruction','http://example.org/Instruction_Parameter','http://book.org/purpose']                           

def clean_pred(o):
    if dataset_name in o:
        o = dataset_name + '-' + o.split('/')[-1] 
    elif 'book' in o:
        o = 'book-' + o.split('/')[-1]
    else:
        o = o.split('/')[-1]
    
    return o


intersection_predicates = [clean_pred(x) for x in intersection_predicates]
intersection_predicates_uri = [clean_pred(x) for x in intersection_predicates_uri]

# print(intersection_predicates)
# print(intersection_predicates_uri)
import rdflib
import re
import collections

literal_len = 30

def dataType(string):
    odp='string'
    patternBIT=re.compile('[01]')
    patternINT=re.compile('[0-9]+')
    patternFLOAT=re.compile('[0-9]+\.[0-9]+')
    patternTEXT=re.compile('[a-zA-Z0-9]+')
    if patternTEXT.match(string):
        odp= "string"
    if patternINT.match(string):
        odp= "integer"
    if patternFLOAT.match(string):
        odp= "float"
    return odp

### Return: data, data_type
def getRDFData(o):
    if isinstance(o, rdflib.term.URIRef):
        data_type = "uri"
    else:
        data_type = o.datatype
        if data_type == None:
            data_type = dataType(o)
        else:
            if "#" in o.datatype:
                data_type = o.datatype.split('#')[1].lower()
            else:
                data_type = dataType(o)
        if data_type == 'gmonthday' or data_type=='gyear':
            data_type = 'date'
        if data_type == 'positiveinteger' or data_type == 'int' or data_type == 'nonnegativeinteger':
            data_type = 'integer'
    return o, data_type


def getRDFData_predicate(s, o):
    o = URIRef(o)
    if isinstance(o, rdflib.term.URIRef):
        data_type = "uri"
    else:
        data_type = o.datatype
        if data_type == None:
            data_type = dataType(o)
        else:
            if "#" in o.datatype:
                data_type = o.datatype.split('#')[1].lower()
            else:
                data_type = dataType(o)
        if data_type == 'gmonthday' or data_type=='gyear':
            data_type = 'date'
        if data_type == 'positiveinteger' or data_type == 'int' or data_type == 'nonnegativeinteger':
            data_type = 'integer'
    
    if dataset_name in o:
        o = dataset_name + '-' + o.split('/')[-1]
    else:
        o = 'book-' + o.split('/')[-1]
    return o, data_type

def getLiteralArray(o, literal_len, char_vocab):
    literal_object = list()
    for i in range(literal_len):
        literal_object.append(0)
    if o[1] != 'uri':
        max_len = min(literal_len, len(o[0]))
        for i in range(max_len):
            if char_vocab.get(o[0][i]) == None:
                char_vocab[o[0][i]] = len(char_vocab)
            literal_object[i] = char_vocab[o[0][i]]
    elif entity_label_dict.get(o[0]) != None:
        label = entity_label_dict.get(o[0])
        max_len = min(literal_len, len(label))
        for i in range(max_len):
            if char_vocab.get(label[i]) == None:
                char_vocab[label[i]] = len(char_vocab)
            literal_object[i] = char_vocab[label[i]]
        
    return literal_object



entity_vocab = dict()
entity_dbp_vocab = list()
entity_dbp_vocab_neg = list()
entity_lgd_vocab_neg = list()
predicate_vocab = dict()
predicate_vocab['<NONE>'] = 0
entity_literal_vocab = dict()
entity_literal_dbp_vocab_neg = list()
entity_literal_lgd_vocab_neg = list()
data_uri = [] ###[ [[s,p,o,p_trans],[chars],predicate_weight], ... ]
data_uri_0 = []
data_literal_0 = []
data_literal = []
data_uri_trans = []
data_literal_trans = []
char_vocab = dict()
char_vocab['<pad>'] = 0
#tmp_data = []

pred_weight = dict()
num_triples = 0
for s, p, o in graph:

    num_triples += 1
    s = getRDFData(s)
    p = getRDFData_predicate(s, p)
    o = getRDFData(o)
    
    if pred_weight.get(p[0]) == None:
        pred_weight[p[0]] = 1
    else:
        pred_weight[p[0]] += 1

    ## all vocab for finding neg sample
    if entity_literal_vocab.get(s[0]) == None:
        entity_literal_vocab[s[0]] = len(entity_literal_vocab)
        if str(s[0]).startswith('http://book.org/'):
            entity_literal_dbp_vocab_neg.append(s[0])
        else:
            entity_literal_lgd_vocab_neg.append(s[0])
    if entity_literal_vocab.get(o[0]) == None:
        entity_literal_vocab[o[0]] = len(entity_literal_vocab)
        if str(s[0]).startswith('http://book.org/'):
            entity_literal_dbp_vocab_neg.append(o[0])
        else:
            entity_literal_lgd_vocab_neg.append(o[0])
        
    if entity_vocab.get(s[0]) == None:
        idx = len(entity_vocab)
        entity_vocab[s[0]] = idx
        if str(s[0]).startswith('http://book.org/'):
            entity_dbp_vocab.append(idx)
            entity_dbp_vocab_neg.append(s[0])
        else:
            entity_lgd_vocab_neg.append(s[0])
    if predicate_vocab.get(p[0]) == None:
        predicate_vocab[p[0]] = len(predicate_vocab)
    if o[1] == 'uri':
        if entity_vocab.get(o[0]) == None:
            entity_vocab[o[0]] = len(entity_vocab)
            if str(s[0]).startswith('http://book.org/'):
                entity_dbp_vocab_neg.append(o[0])
            else:
                entity_lgd_vocab_neg.append(o[0])
        literal_object = getLiteralArray(o, literal_len, char_vocab)
        if str(p[0]) not in intersection_predicates_uri:
            data_uri_0.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o[0]], 0], literal_object])
        else:
            data_uri.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o[0]], 0], literal_object])
            ### DATA TRANS
            duplicate_preds = [item for item, count in collections.Counter(graph.predicates(o[0],None)).items() if count > 1]
            if True:
              for g1 in graph.triples((o[0],None,None)):
                  if len(g1) > 0:
                      s1,p1,o1 = g1
                      s1 = getRDFData(s1)
                      p1 = getRDFData_predicate(s1, p1)
                      o1 = getRDFData(o1)

                      if entity_vocab.get(o1[0]) == None:
                          entity_vocab[o1[0]] = len(entity_vocab)
                      if str(s1[0]).startswith('http://book.org/'):
                          entity_dbp_vocab_neg.append(o1[0])
                      else:
                          entity_lgd_vocab_neg.append(o1[0])
                      if entity_vocab.get(o1[1]) == None:
                          entity_vocab[o1[1]] = len(entity_vocab)
                      if predicate_vocab.get(p1[0]) == None:
                          predicate_vocab[p1[0]] = len(predicate_vocab)
                      if p[0] != p1[0] and len(set(clean_pred(x) for x in (graph.predicates(s[0]))).intersection(set(intersection_predicates_uri))) != 0:
                          if isinstance(o1[0], rdflib.term.URIRef) and str(p1[0]) in intersection_predicates_uri:
                              data_uri_trans.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o1[0]], predicate_vocab[p1[0]]], getLiteralArray(o1, literal_len, char_vocab)])
                          elif isinstance(o1[0], rdflib.term.Literal) and 'rdf-schema#label' in str(p1[0]):
                              data_literal_trans.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o1[1]], predicate_vocab[p1[0]]], getLiteralArray(o1, literal_len, char_vocab)])
                              #tmp_data.append((s[0], p[0], o[0], p1[0], o1[0]))
              ##############
    else:
        if entity_vocab.get(o[1]) == None:
            entity_vocab[o[1]] = len(entity_vocab)
        literal_object = getLiteralArray(o, literal_len, char_vocab)
        if str(p[0]) not in intersection_predicates:
            data_literal_0.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o[1]], 0], literal_object])
        else:
            data_literal.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o[1]], 0], literal_object])
            

reverse_entity_vocab = invert_dict(entity_vocab)
reverse_predicate_vocab = invert_dict(predicate_vocab)
reverse_char_vocab = invert_dict(char_vocab)
reverse_entity_literal_vocab = invert_dict(entity_literal_vocab)

#Add predicate weight
for i in range(0, len(data_uri)):
    s = reverse_entity_vocab.get(data_uri[i][0][0])
    p = reverse_predicate_vocab.get(data_uri[i][0][1])
    data_uri[i].append([(pred_weight.get(p)/float(num_triples))])

for i in range(0, len(data_uri_0)):
    s = reverse_entity_vocab.get(data_uri_0[i][0][0])
    p = reverse_predicate_vocab.get(data_uri_0[i][0][1])
    data_uri_0[i].append([(pred_weight.get(p)/float(num_triples))])

for i in range(0, len(data_uri_trans)):
    s = reverse_entity_vocab.get(data_uri_trans[i][0][0])
    p = reverse_predicate_vocab.get(data_uri_trans[i][0][1])
    data_uri_trans[i].append([(pred_weight.get(p)/float(num_triples))])
    
for i in range(0, len(data_literal)):
    s = reverse_entity_vocab.get(data_literal[i][0][0])
    p = reverse_predicate_vocab.get(data_literal[i][0][1])
    data_literal[i].append([(pred_weight.get(p)/float(num_triples))])

for i in range(0, len(data_literal_0)):
    s = reverse_entity_vocab.get(data_literal_0[i][0][0])
    p = reverse_predicate_vocab.get(data_literal_0[i][0][1])
    data_literal_0[i].append([(pred_weight.get(p)/float(num_triples))])
    
for i in range(0, len(data_literal_trans)):
    s = reverse_entity_vocab.get(data_literal_trans[i][0][0])
    p = reverse_predicate_vocab.get(data_literal_trans[i][0][1])
    data_literal_trans[i].append([(pred_weight.get(p)/float(num_triples))])
    
if len(data_uri_trans) < 100:
    data_uri_trans = data_uri_trans+data_uri_trans
    
print (len(entity_vocab), len(predicate_vocab), len(char_vocab), len(entity_dbp_vocab))



# predicate proximity triples

ent_type_vocab = dict()
data_predicate = list()
domain_vocab = dict()
range_vocab = dict()
for t in  predicate_graph:
    s,p,o = t
    p = getRDFData_predicate(s,p)[0]
    if s not in ent_type_vocab:
        ent_type_vocab[s] = len(ent_type_vocab)
    if o not in ent_type_vocab:
        ent_type_vocab[o] = len(ent_type_vocab)
    data_predicate.append([[ent_type_vocab[s],predicate_vocab[p],ent_type_vocab[o], 0], [0]*literal_len, [0]])
    
    if predicate_vocab[p] in domain_vocab:
        domain_vocab[predicate_vocab[p]].add(ent_type_vocab[s])
    else:
        domain_vocab[predicate_vocab[p]] = set()
        domain_vocab[predicate_vocab[p]].add(ent_type_vocab[s])
        
    if predicate_vocab[p] in range_vocab:
        range_vocab[predicate_vocab[p]].add(ent_type_vocab[o])
    else:
        range_vocab[predicate_vocab[p]] = set()
        range_vocab[predicate_vocab[p]].add(ent_type_vocab[o])


print (len(data_uri_trans))
print (len(data_literal_trans))
print (len(data_uri))
print (len(data_literal))
print (len(data_uri_0))
print (len(data_literal_0))
print (len(data_predicate))



def getBatch(data, batchSize, current, entityVocab, literal_len, char_vocab):
    hasNext = current+batchSize < len(data)
    
    if (len(data) - current) < batchSize:
        current = current - (batchSize - (len(data) - current))
    
    dataPos_all = data[current:current+batchSize]
    dataPos = list()
    charPos = list()
    pred_weight_pos = list()
    dataNeg = list()
    predNeg = list()
    predTransNeg = list()
    charNeg = list()
    pred_weight_neg = list()
    for triples,chars, pred_weight in dataPos_all:
        s,p,o,p_trans = triples
        dataPos.append([s,p,o,p_trans])
        charPos.append(chars)
        pred_weight_pos.append(pred_weight)
        lr = round(random.random())
        
        if lr == 0:
            try:
                o_type = getRDFData(reverse_entity_vocab[o])
            except:
                o_type = 'not_uri'
            
            literal_array = []
            rerun = True
            while rerun or negElm[0] == (reverse_entity_vocab[o] and literal_array == chars):
                if o_type[1] == 'uri':
                    if str(s).startswith('http://book.org/'):
                        negElm = entity_dbp_vocab_neg[random.randint(0, len(entity_dbp_vocab_neg)-1)]
                        negElm = reverse_entity_vocab[entity_vocab[negElm]]
                    else:
                        negElm = entity_lgd_vocab_neg[random.randint(0, len(entity_lgd_vocab_neg)-1)]
                        negElm = reverse_entity_vocab[entity_vocab[negElm]]
                else:
                    if str(s).startswith('http://book.org/'):
                        negElm = entity_literal_dbp_vocab_neg[random.randint(0, len(entity_literal_dbp_vocab_neg)-1)]
                        negElm = reverse_entity_literal_vocab[entity_literal_vocab[negElm]]
                    else:
                        negElm = entity_literal_lgd_vocab_neg[random.randint(0, len(entity_literal_lgd_vocab_neg)-1)]
                        negElm = reverse_entity_literal_vocab[entity_literal_vocab[negElm]]
                if o_type == 'uri' and negElm[1] == 'uri':
                    rerun = False
                elif o_type != 'uri':
                    rerun = False
                if (isinstance(negElm, rdflib.term.URIRef)) or (isinstance(negElm, rdflib.term.Literal)):
                    negElm = getRDFData(negElm)
                    literal_array = getLiteralArray(negElm, literal_len, char_vocab)
                else:
                    rerun = True    
            if negElm[1] == 'uri':
                dataNeg.append([s, p, entity_vocab[negElm[0]], p_trans])
            else:
                dataNeg.append([s, p, entity_vocab[negElm[1]], p_trans])
            charNeg.append(literal_array)
            predNeg.append(p)
            pred_weight_neg.append(pred_weight)
        else:
            ### SUBJECTS only contains URI
            negElm = random.randint(0, len(entity_vocab)-1)
            while negElm == s:
                negElm = random.randint(0, len(entity_vocab)-1)
            dataNeg.append([negElm, p, o, p_trans])
            charNeg.append(chars)
            predNeg.append(p)
            pred_weight_neg.append(pred_weight)
            
    dataPos = np.array(dataPos)
    charPos = np.array(charPos)
    pred_weight_pos = np.array(pred_weight_pos)
    dataNeg = np.array(dataNeg)
    predNeg = np.array(predNeg)
    predTransNeg = np.array(predTransNeg)
    charNeg = np.array(charNeg)
    pred_weight_neg = np.array(pred_weight_neg)
    
    return hasNext, current+batchSize, dataPos[:,0], dataPos[:,1], dataPos[:,2], dataPos[:,3], pred_weight_pos, charPos, dataNeg[:,0], dataNeg[:,1], dataNeg[:,2], dataNeg[:,3], pred_weight_neg, charNeg
    #return charNeg   



def getBatchPP(data, batchSize, current, entityVocab, literal_len, char_vocab):
    hasNext = current+batchSize < len(data)
    
    if (len(data) - current) < batchSize:
        current = current - (batchSize - (len(data) - current))
    
    dataPos_all = data[current:current+batchSize]
    dataPos = list()
    charPos = list()
    pred_weight_pos = list()
    dataNeg = list()
    charNeg = list()
    pred_weight_neg = list()
    for triples,chars, pred_weight in dataPos_all:
        s,p,o,p_trans = triples
        print (s,p,o)
        print()
        dataPos.append([s,p,o,p_trans])
        charPos.append(chars)
        pred_weight_pos.append(pred_weight)
        charNeg.append(chars)
        pred_weight_neg.append(pred_weight)
        lr = round(random.random())
        
        if lr == 0: #randomize object
            candidate = set(ent_type_vocab.values()) - range_vocab[p]
            o_rand = random.choice(list(candidate))
            dataNeg.append([s,p,o_rand,p_trans])
        else: #randomize object
            candidate = set(ent_type_vocab.values()) - domain_vocab[p]
            s_rand = random.choice(list(candidate))
            dataNeg.append([s_rand,p,o,p_trans])
            
    dataPos = np.array(dataPos)
    charPos = np.array(charPos)
    pred_weight_pos = np.array(pred_weight_pos)
    dataNeg = np.array(dataNeg)
    charNeg = np.array(charNeg)
    pred_weight_neg = np.array(pred_weight_neg)
    
    return hasNext, current+batchSize, dataPos[:,0], dataPos[:,1], dataPos[:,2], dataPos[:,3], pred_weight_pos, charPos, dataNeg[:,0], dataNeg[:,1], dataNeg[:,2], dataNeg[:,3], pred_weight_neg, charNeg
    #return charNeg   



batchSize = 100
hidden_size = 100
totalEpoch = 400
verbose = 1000
margin = 1.0
literal_len = literal_len
entitySize = len(entity_vocab)
charSize = len(char_vocab)
predSize = len(predicate_vocab)
ppEntSize = len(ent_type_vocab)
valid_size = 100 #100 entity validation sample
top_k = 5


import random
from rdflib import URIRef

file_lgd = open(map_file, 'r')

valid_dataset_list = list()
for line in file_lgd:
    elements = line.split(' ')
    s = elements[0] #DBpedia
    p = elements[1]
    o = elements[2] #LGD
    
    if (entity_vocab[URIRef(s.replace('<','').replace('>',''))] in entity_dbp_vocab) and (URIRef(o.replace('<','').replace('>','')) in entity_vocab):
        valid_dataset_list.append((o, s))
#valid_dataset_list = random.sample(valid_dataset_list, valid_size)
file_lgd.close()
print(valid_dataset_list)
valid_examples = [entity_vocab[URIRef(k.replace('<','').replace('>',''))] for k,_ in valid_dataset_list] #LGD
valid_answer = [entity_dbp_vocab.index(entity_vocab[URIRef(k.replace('<','').replace('>',''))]) for _,k in valid_dataset_list] #DBpedia
print(valid_answer)
print(valid_examples)

from tensorflow.contrib import rnn

tfgraph = tf.Graph()

with tfgraph.as_default():
    
    # input placeholder #
    pos_h = tf.placeholder(tf.int32, [None])
    pos_t = tf.placeholder(tf.int32, [None])
    pos_r = tf.placeholder(tf.int32, [None])
    pos_r_trans = tf.placeholder(tf.int32, [None])
    pos_c = tf.placeholder(tf.int32, [None, literal_len])
    pos_pred_weight = tf.placeholder(tf.float32, [None,1], name='pos_pred_weight')

    neg_h = tf.placeholder(tf.int32, [None])
    neg_t = tf.placeholder(tf.int32, [None])
    neg_r = tf.placeholder(tf.int32, [None])
    neg_r_trans = tf.placeholder(tf.int32, [None])
    neg_c = tf.placeholder(tf.int32, [None, literal_len])
    neg_pred_weight = tf.placeholder(tf.float32, [None,1], name='neg_pred_weight')

    pos_h_type = tf.placeholder(tf.int32, [None,4]) # [b, 4]
    pos_t_type = tf.placeholder(tf.int32, [None,4]) # [b, 4]
    neg_h_type = tf.placeholder(tf.int32, [None,4]) # [b, 4]
    neg_t_type = tf.placeholder(tf.int32, [None,4]) # [b, 4]
    
    type_data = tf.placeholder(tf.int32, [1])
    type_trans = tf.placeholder(tf.int32, [1])
    ######################
    
    # embedding variables #
    ent_embeddings_ori = tf.get_variable(name = "relationship_triple_ent_embedding", shape = [entitySize, hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    atr_embeddings_ori = tf.get_variable(name = "attribute_triple_ent_embedding", shape = [entitySize, hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    ent_rel_embeddings = tf.get_variable(name = "proximity_triple_pred_embedding", shape = [predSize, hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    pp_ent_embeddings = tf.get_variable(name = "proximity_triple_ent_embedding", shape = [ppEntSize, hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    #atr_rel_embeddings = tf.get_variable(name = "attribute_triple_pred_embedding", shape = [predSize, hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    char_embeddings = tf.get_variable(name = "attribute_triple_char_embedding", shape = [charSize, hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    pp_w_embeddings = tf.get_variable(name = "proximity_triple_weight_embedding", shape = [hidden_size, hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    
    ent_indices = tf.concat([pos_h, pos_t, neg_h, neg_t], 0)
    ent_indices = tf.reshape(ent_indices,[-1,1])
    ent_value = tf.concat([tf.nn.embedding_lookup(ent_embeddings_ori, pos_h),                          tf.nn.embedding_lookup(ent_embeddings_ori, pos_t),                          tf.nn.embedding_lookup(ent_embeddings_ori, neg_h),                          tf.nn.embedding_lookup(ent_embeddings_ori, neg_t)], 0)
    part_ent_embeddings = tf.scatter_nd([ent_indices], [ent_value], ent_embeddings_ori.shape)
    ent_embeddings = part_ent_embeddings + tf.stop_gradient(-part_ent_embeddings + ent_embeddings_ori)
    
    atr_indices = tf.concat([pos_h, pos_t, neg_h, neg_t], 0)
    atr_indices = tf.reshape(atr_indices,[-1,1])
    atr_value = tf.concat([tf.nn.embedding_lookup(atr_embeddings_ori, pos_h),                          tf.nn.embedding_lookup(atr_embeddings_ori, pos_t),                          tf.nn.embedding_lookup(atr_embeddings_ori, neg_h),                          tf.nn.embedding_lookup(atr_embeddings_ori, neg_t)], 0)
    part_atr_embeddings = tf.scatter_nd([atr_indices], [atr_value], atr_embeddings_ori.shape)
    atr_embeddings = part_atr_embeddings + tf.stop_gradient(-part_atr_embeddings + atr_embeddings_ori)
    ########################
    
    
    # PREDICATE PROXIMITY TRIPLES #
    if aggregation == 'W':
        pp_pos_h_e = tf.nn.embedding_lookup(pp_ent_embeddings, pos_h)
        pp_pos_t_e = tf.nn.embedding_lookup(pp_ent_embeddings, pos_t)
        pp_pos_r_e = tf.nn.embedding_lookup(ent_rel_embeddings, pos_r)
        pp_neg_h_e = tf.nn.embedding_lookup(pp_ent_embeddings, neg_h)
        pp_neg_t_e = tf.nn.embedding_lookup(pp_ent_embeddings, neg_t)
        pp_neg_r_e = tf.nn.embedding_lookup(ent_rel_embeddings, neg_r)
        pp_pos = tf.reduce_sum(abs(pp_pos_h_e + pp_pos_r_e - pp_pos_t_e), 1, keep_dims = True)
        pp_neg = tf.reduce_sum(abs(pp_neg_h_e + pp_neg_r_e - pp_neg_t_e), 1, keep_dims = True)
        #pp_learning_rate = 0.0001 # LGD/GEO
        pp_learning_rate = 0.0001 # YAGO
        pp_opt_vars_ent = [v for v in tf.trainable_variables() if v.name.startswith("proximity_triple")]
        pp_loss = tf.reduce_sum(tf.maximum(pp_pos - pp_neg + 1, 0)) # LGD/GEO
        # pp_loss = tf.reduce_sum(tf.maximum(pp_pos - pp_neg + 10, 0)) # YAGO
        pp_optimizer = tf.train.AdamOptimizer(pp_learning_rate).minimize(pp_loss, var_list=pp_opt_vars_ent)
    elif aggregation == 'A':
        # pos_h mapping from domain_vocab, 
        pp_pos_h_e = tf.nn.embedding_lookup(pp_ent_embeddings, pos_h_type) # [b, 4, hidden_size]
        # tf.matmul() only for 2 dims
        pp_pos_h_e =  tf.reshape(pp_pos_h_e, [-1, hidden_size])  # [b*4, hid]
        tmp = tf.matmul(pp_pos_h_e, pp_w_embeddings)  # [b*4, hid]
        tmp = tf.reshape(tmp, [-1, 4, hidden_size]) 
        att_w = tf.nn.softmax(tf.reduce_sum(tmp, -1, keep_dims=True), axis=-1) # [b, 4, 1]
        pp_pos_h_e = tf.reduce_sum(tf.multiply(att_w, tmp), axis=1) # [b, hidden_size]
        pp_pos_r_e = tf.nn.embedding_lookup(ent_rel_embeddings, pos_r)
        pp_neg_r_e = tf.nn.embedding_lookup(ent_rel_embeddings, neg_r)

        # pp_pos_t_e
        pp_pos_t_e = tf.nn.embedding_lookup(pp_ent_embeddings, pos_t_type)
        tmp = tf.reshape(tf.matmul(tf.reshape(pp_pos_t_e, [-1, hidden_size]), pp_w_embeddings), [-1, 4, hidden_size])
        att_w = tf.nn.softmax(tf.reduce_sum(tmp, -1, keep_dims=True), axis=-1) # [b, 4, 1]
        pp_pos_t_e = tf.reduce_sum(tf.multiply(att_w, tmp), axis=1) # [b, hidden_size]
        
        # pp_neg_h_e
        pp_neg_h_e = tf.nn.embedding_lookup(pp_ent_embeddings, neg_h_type)
        tmp = tf.reshape(tf.matmul(tf.reshape(pp_neg_h_e, [-1, hidden_size]), pp_w_embeddings), [-1, 4, hidden_size])
        att_w = tf.nn.softmax(tf.reduce_sum(tmp, -1, keep_dims=True), axis=-1) # [b, 4, 1]
        pp_neg_h_e = tf.reduce_sum(tf.multiply(att_w, tmp), axis=1) # [b, hidden_size]

        # pp_neg_t_e
        pp_neg_t_e = tf.nn.embedding_lookup(pp_ent_embeddings, neg_t_type)
        tmp = tf.reshape(tf.matmul(tf.reshape(pp_neg_t_e, [-1, hidden_size]), pp_w_embeddings), [-1, 4, hidden_size])
        att_w = tf.nn.softmax(tf.reduce_sum(tmp, -1, keep_dims=True), axis=-1) # [b, 4, 1]
        pp_neg_t_e = tf.reduce_sum(tf.multiply(att_w, tmp), axis=1) # [b, hidden_size]

        pp_pos = tf.reduce_sum(abs(pp_pos_h_e + pp_pos_r_e - pp_pos_t_e), 1, keep_dims = True)
        pp_neg = tf.reduce_sum(abs(pp_neg_h_e + pp_neg_r_e - pp_neg_t_e), 1, keep_dims = True)
        #pp_learning_rate = 0.0001 # LGD/GEO
        pp_learning_rate = 0.0001 # YAGO
        pp_opt_vars_ent = [v for v in tf.trainable_variables() if v.name.startswith("proximity_triple")]
        #pp_loss = tf.reduce_sum(tf.maximum(pp_pos - pp_neg + 1, 0)) # LGD/GEO
        pp_loss = tf.reduce_sum(tf.maximum(pp_pos - pp_neg + 10, 0)) # YAGO
        pp_optimizer = tf.train.AdamOptimizer(pp_learning_rate).minimize(pp_loss, var_list=pp_opt_vars_ent)
    ########################
    
    
    # RELATIONSHIP TRIPLES #
    rt_pos_h_e = tf.nn.embedding_lookup(ent_embeddings, pos_h)
    rt_pos_t_e = tf.nn.embedding_lookup(ent_embeddings, pos_t)
    #rt_pos_r_e = tf.nn.embedding_lookup(ent_rel_embeddings, pos_r) # LGD/GEO
    rt_pos_r_e = tf.stop_gradient(tf.nn.embedding_lookup(ent_rel_embeddings, pos_r)) # YAGO
    rt_neg_h_e = tf.nn.embedding_lookup(ent_embeddings, neg_h)
    rt_neg_t_e = tf.nn.embedding_lookup(ent_embeddings, neg_t)
    #rt_neg_r_e = tf.nn.embedding_lookup(ent_rel_embeddings, neg_r) # LGD/GEO
    rt_neg_r_e = tf.stop_gradient(tf.nn.embedding_lookup(ent_rel_embeddings, neg_r)) # YAGO
    
    ent_emb = tf.nn.embedding_lookup(ent_embeddings, pos_h)
    atr_emb = tf.nn.embedding_lookup(atr_embeddings, pos_h)
    norm_ent_emb = tf.nn.l2_normalize(ent_emb,1)
    norm_atr_emb = tf.nn.l2_normalize(atr_emb,1)
    cos_sim = tf.reduce_sum(tf.multiply(norm_ent_emb, norm_atr_emb), 1, keep_dims=True)
    
    rt_pos = tf.reduce_sum(abs(rt_pos_h_e + rt_pos_r_e - rt_pos_t_e), 1, keep_dims = True)
    rt_neg = tf.reduce_sum(abs(rt_neg_h_e + rt_neg_r_e - rt_neg_t_e), 1, keep_dims = True)
    #rt_learning_rate = tf.reduce_min(pos_pred_weight)*0.001 # LGD/GEO
    rt_learning_rate = 0.0001 # YAGO
    rt_opt_vars_ent = [v for v in tf.trainable_variables() if v.name.startswith("relationship_triple")]
    #rt_loss = tf.reduce_sum(tf.maximum(rt_pos - rt_neg + 1, 0)) # LGD/GEO
    rt_loss = tf.reduce_sum(tf.maximum(rt_pos - rt_neg + 10, 0)) # YAGO
    rt_optimizer = tf.train.AdamOptimizer(rt_learning_rate).minimize(rt_loss, var_list=rt_opt_vars_ent)
    ########################
                            
                            
                            
    # ATTRIBUTE TRIPLES #
    at_pos_h_e = tf.nn.embedding_lookup(atr_embeddings, pos_h)
    pos_c_e = tf.nn.embedding_lookup(char_embeddings, pos_c)
    at_pos_r_e = tf.stop_gradient(tf.nn.embedding_lookup(ent_rel_embeddings, pos_r))
    #at_pos_r_e = tf.nn.embedding_lookup(atr_rel_embeddings, pos_r)
    at_neg_h_e = tf.nn.embedding_lookup(atr_embeddings, neg_h)
    neg_c_e = tf.nn.embedding_lookup(char_embeddings, neg_c)
    at_neg_r_e = tf.stop_gradient(tf.nn.embedding_lookup(ent_rel_embeddings, neg_r))
    #at_neg_r_e = tf.nn.embedding_lookup(atr_rel_embeddings, neg_r)
    
    #Zero-Mask for char embedding
    mask_constant_0 = np.zeros([1,hidden_size])
    mask_constant_1 = np.ones([1,hidden_size])
    mask_constant = np.concatenate([mask_constant_0, mask_constant_1])
    mask_constant = tf.constant(mask_constant, tf.float32)
    flag_pos_c_e = tf.sign(tf.abs(pos_c))
    mask_pos_c_e = tf.nn.embedding_lookup(mask_constant, flag_pos_c_e)
    pos_c_e = pos_c_e * mask_pos_c_e
    flag_neg_c_e = tf.sign(tf.abs(neg_c))
    mask_neg_c_e = tf.nn.embedding_lookup(mask_constant, flag_neg_c_e)
    neg_c_e = neg_c_e * mask_neg_c_e
    
    
    #N-GRAM
    def calculate_ngram_weight(unstacked_tensor):
        stacked_tensor = tf.stack(unstacked_tensor, 1)
        stacked_tensor = tf.reverse(stacked_tensor, [1])
        index = tf.constant(len(unstacked_tensor))
        expected_result = tf.zeros([batchSize, hidden_size])
        def condition(index, summation):
            return tf.greater(index, 0)
        def body(index, summation):
            precessed = tf.slice(stacked_tensor,[0,index-1,0], [-1,-1,-1])
            summand = tf.reduce_mean(precessed, 1)
            return tf.subtract(index, 1), tf.add(summation, summand)
        result = tf.while_loop(condition, body, [index, expected_result])
        return result[1]
    pos_c_e_in_lstm = tf.unstack(pos_c_e, literal_len, 1)
    pos_c_e_lstm = calculate_ngram_weight(pos_c_e_in_lstm)
    neg_c_e_in_lstm = tf.unstack(neg_c_e, literal_len, 1)
    neg_c_e_lstm = calculate_ngram_weight(neg_c_e_in_lstm)
    
    at_pos = tf.reduce_sum(abs(at_pos_h_e + at_pos_r_e - pos_c_e_lstm), 1, keep_dims = True)
    at_neg = tf.reduce_sum(abs(at_neg_h_e + at_neg_r_e - neg_c_e_lstm), 1, keep_dims = True)
    at_pos_h_e = tf.multiply(at_pos, pos_pred_weight)
    at_neg_h_e = tf.multiply(at_neg, neg_pred_weight)
    #at_learning_rate = tf.reduce_min(pos_pred_weight)*0.001 # LGD/GEO
    at_learning_rate = tf.reduce_min(pos_pred_weight)*0.01 # YAGO
    at_opt_vars_atr = [v for v in tf.trainable_variables() if v.name.startswith("attribute_triple") or v.name.startswith("rnn")]
    at_loss = tf.reduce_sum(tf.maximum(at_pos - at_neg + 1, 0))
    at_optimizer = tf.train.AdamOptimizer(at_learning_rate).minimize(at_loss, var_list=at_opt_vars_atr)
    ########################
                            
    
    # TRANSITIVE TRIPLES #
    pos_r_e_trans = tf.stop_gradient(tf.nn.embedding_lookup(ent_rel_embeddings, pos_r_trans))
    neg_r_e_trans = tf.stop_gradient(tf.nn.embedding_lookup(ent_rel_embeddings, neg_r_trans))
    tr_pos_r_e = tf.multiply(at_pos_r_e, pos_r_e_trans)
    tr_neg_r_e = tf.multiply(at_neg_r_e, neg_r_e_trans)
    tr_pos = tf.reduce_sum(abs(at_pos_h_e + tr_pos_r_e - pos_c_e_lstm), 1, keep_dims = True)
    tr_neg = tf.reduce_sum(abs(at_neg_h_e + tr_neg_r_e - neg_c_e_lstm), 1, keep_dims = True)
    tr_pos_h_e = tf.multiply(tr_pos, pos_pred_weight)
    tr_neg_h_e = tf.multiply(tr_neg, neg_pred_weight)
    #tr_learning_rate = tf.reduce_min(pos_pred_weight)*0.001 # LGD/GEO
    tr_learning_rate = tf.reduce_min(pos_pred_weight)*0.01 # YAGO
    tr_opt_vars_atr = [v for v in tf.trainable_variables() if v.name.startswith("attribute_triple") or v.name.startswith("rnn")]
    tr_loss = tf.reduce_sum(tf.maximum(tr_pos - tr_neg + 1, 0))
    tr_optimizer = tf.train.AdamOptimizer(tr_learning_rate).minimize(tr_loss, var_list=tr_opt_vars_atr)
    ######################
    
    
    #Entity Embeddings & Attribute Embeddings Similarity
    sim_ent_emb = tf.nn.embedding_lookup(ent_embeddings, pos_h)
    sim_atr_emb = tf.nn.embedding_lookup(atr_embeddings, pos_h)
    norm_ent_emb = tf.nn.l2_normalize(sim_ent_emb,1)
    norm_atr_emb = tf.nn.l2_normalize(sim_atr_emb,1)
    cos_sim = tf.reduce_sum(tf.multiply(norm_ent_emb, norm_atr_emb), 1, keep_dims=True)
    sim_loss = tf.reduce_sum(1-cos_sim)
    opt_vars_sim = [v for v in tf.trainable_variables() if v.name.startswith("relationship_triple_ent_embedding")]
    sim_optimizer = tf.train.AdamOptimizer(0.01).minimize(sim_loss, var_list=opt_vars_sim)
    ####################################################
    
    
                            
    # testing
    with tf.device('/cpu:0'):
        norm = tf.sqrt(tf.reduce_sum(tf.square(ent_embeddings_ori), 1, keep_dims=True))
        normalized_embeddings = ent_embeddings_ori / norm

        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
    init = tf.global_variables_initializer()
    #########


def metric(y_true, y_pred, answer_vocab, k=10):
    list_rank = list()
    total_hits = 0
    total_hits_1 = 0
    for i in range(len(y_true)):
        result = y_pred[i]
        result = result[answer_vocab]
        #sort result first
        result = (-result).argsort()
        
        #Mean Rank
        for j in range(len(result)):
            if result[j] == y_true[i]:
                rank = j
                break
        list_rank.append(j)
        #Mean Rank
        
        #Hit @K
        result = result[:k]
        for j in range(len(result)):
            if result[j] == y_true[i]:
                total_hits += 1
                if j == 0:
                    total_hits_1 += 1
                break
    
    #RETURN: MeanRank, Hits@K
    return reduce(lambda x, y: x + y, list_rank) / len(list_rank), float(total_hits)/len(y_true), float(total_hits_1)/len(y_true)


def run(graph, totalEpoch):
    writer = open('log.txt', 'w')
    with tf.Session(graph=graph) as session:
        writer_tb=tf.summary.FileWriter('./tensorboard_study', session.graph)
        init.run()
        
        for epoch in range(totalEpoch):
            if epoch % 2 == 0:
                data = [data_predicate, data_uri_0, data_uri, data_literal_0, data_literal, []]
            else:
                data = [data_literal]
            start_time_epoch = dt.datetime.now()
            for i in range(0, len(data)):
                random.shuffle(data[i])
                hasNext = True
                current = 0
                step = 0
                average_loss = 0
                t_start = time.time()  
                while(hasNext and len(data[i]) > 0):
                    step += 1
                    if epoch % 2 == 0 and i == 0:
                        hasNext, current, ph, pr, pt, pr_trans, ppred, pc, nh, nr, nt, nr_trans, npred, nc = getBatch(data[i], batchSize, current, entity_vocab, literal_len, char_vocab)
                    else:
                        hasNext, current, ph, pr, pt, pr_trans, ppred, pc, nh, nr, nt, nr_trans, npred, nc = getBatch(data[i], batchSize, current, entity_vocab, literal_len, char_vocab)
                    feed_dict = {
                        pos_h: ph,
                        pos_t: pt,
                        pos_r: pr,
                        pos_r_trans: pr_trans,
                        pos_pred_weight : ppred,
                        pos_c: pc,
                        neg_h: nh,
                        neg_t: nt,
                        neg_r: nr,
                        neg_r_trans: nr_trans,
                        neg_c: nc,
                        neg_pred_weight: npred,
                    }
                    # compute entity embedding and attribute embedding
                    
                    if epoch % 2 == 0:
                        if i == 0: # predicate proximity triples
                            __, loss_val = session.run([pp_optimizer, pp_loss], feed_dict=feed_dict)
                            
                            average_loss += loss_val
                        elif i == 1 or i == 2: # relationship triples
                            __, loss_val = session.run([rt_optimizer, rt_loss], feed_dict=feed_dict)
                            
                            average_loss += loss_val
                        elif i == 3 or i == 4: # attribute triples
                            __, loss_val = session.run([at_optimizer, at_loss], feed_dict=feed_dict)
                            
                            average_loss += loss_val
                        elif i == 5: # transitive triples
                            __, loss_val = session.run([tr_optimizer, tr_loss], feed_dict=feed_dict)
                            
                            average_loss += loss_val
                    # compute entity embedding similarity
                    else:
                        __, loss_val = session.run([sim_optimizer, sim_loss], feed_dict=feed_dict)
                        
                        average_loss += loss_val

                    if step % verbose == 0:
                        average_loss /= verbose
                        print('Epoch: ', epoch, ' Average loss at step ', step, ': ', average_loss)
                        writer.write('Epoch: '+ str(epoch)+ ' Average loss at step '+ str(step)+ ': '+ str(average_loss)+'\n')
                        average_loss = 0
                
                if len(data[i]) > 0:
                        average_loss /= ((len(data[i])%(verbose*batchSize))/batchSize)
                        print('Epoch: ', epoch, ' Average loss at step ', step, ': ', average_loss)
                        writer.write('Epoch: '+ str(epoch)+ ' Average loss at step '+ str(step)+ ': '+ str(average_loss)+ '\n')
            end_time_epoch = dt.datetime.now()
            print("Training time took {} seconds to run 1 epoch".format((end_time_epoch-start_time_epoch).total_seconds()))
            if (epoch) % 10 == 0:
                start_time_epoch = dt.datetime.now()
                sim = similarity.eval()
                mean_rank, hits_at_10, hits_at_1 = metric(valid_answer, sim, entity_dbp_vocab, top_k)
                print ("Mean Rank: ", mean_rank, " of ", len(entity_dbp_vocab))
                print ("Hits @ "+str(top_k)+": ", hits_at_10)
                print ("Hits @ "+str(1)+": ", hits_at_1)
                end_time_epoch = dt.datetime.now()
                print("Testing time took {} seconds.".format((end_time_epoch-start_time_epoch).total_seconds()))
                print()
            #break
        final_embeddings_normalized = normalized_embeddings.eval()
        final_embeddings_entity = ent_embeddings_ori.eval()
        final_embeddings_predicate = ent_rel_embeddings.eval()



start_time = dt.datetime.now()
run(tfgraph, 401) 
end_time = dt.datetime.now()
print("Training time took {} seconds to run {} epoch".format((end_time-start_time).total_seconds(), totalEpoch))

