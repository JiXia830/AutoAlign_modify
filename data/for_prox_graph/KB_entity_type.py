from SPARQLWrapper import SPARQLWrapper
from rdflib import Graph
import _pickle as cPickle
import pandas as pd
import rdflib
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'

# dy_filename = "./../DY-NB/dbp_yago.ttl"
# dy_filename = "./../DY-NB/test.ttl"
dy_filename = "./../DY-NB/SPO_book_eng_test.ttl"

prox_graph_file = "./SPO_pred_prox_graph1"
graph = Graph()
graph.parse(location=dy_filename, format='nt')
print("len(graph):", len(graph))




def getRdfType(Q):
    Q_types = []

    queryString = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbr: <http://dbpedia.org/resource>
    PREFIX dbo: <http://dbpedia.org/ontology>
    SELECT DISTINCT ?obj WHERE{
    """+ Q +""" rdf:type ?obj
    FILTER strstarts(str(?obj), str(dbo:))
    }
    """

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    sparql.setQuery(queryString)  
    sparql.setTimeout(1000)

    sparql.setReturnFormat("json")

    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            Q_types.append(result["obj"]["value"].replace("http://dbpedia.org/ontology/",""))
        return Q_types
    except TimeoutError:
        return []

def dataType(string):
    odp='string'
    patternBIT=re.compile('[01]')
    patternINT=re.compile('[0-9]+')
    # patternFLOAT=re.compile('[0-9]+\.[0-9]+')
    patternFLOAT=re.compile(r'[0-9]+\.[0-9]+[a-zA-Z]*')
    patternTEXT=re.compile('[a-zA-Z0-9]+')
    patternDate=re.compile('(\d{4})-(\d{2})-(\d{2})')
    if patternTEXT.match(string):
        odp= "string"
    if patternINT.match(string):
        odp= "integer"
    if patternFLOAT.match(string):
        odp= "float"
    if patternDate.match(string):
        odp= "date"
    return odp


def getRDFData(o):
    if str(o).startswith('http://dbpedia.org/resource/'):
        Q_entity = "<"+o+">"
        data_type = getRdfType(Q_entity)
    else:
        data_type = [dataType(o)]
    
    return o, data_type


def add_to_set(types, typeset):
    for t in types:
        typeset.add(t)

typeset1 = set()
typeset2 = set()

prox_graph = []
i=0
for s,p,o in graph:
    i += 1
    # print(i)
    s, s_data_type = getRDFData(str(s)) # change data type
    o, o_data_type = getRDFData(str(o))
    # print(o, o_data_type)
    # print(s, p, o)
    # print(s_data_type, o_data_type)
    add_to_set(s_data_type, typeset1)
    add_to_set(o_data_type, typeset2)
    
    prox_triple_list = [','.join(s_data_type), p, ','.join(o_data_type)]
    print("prox_triple_list:-----------------", prox_triple_list)
    prox_triple_string = '\t'.join(prox_triple_list)
    print("prox_triple_string:---------------", prox_triple_string)
    prox_graph.append(prox_triple_string)
    print("prox_graph------------------------------", prox_graph)
    if i % 10 == 0:
        with open(f"{prox_graph_file}.txt", 'a+') as f:
            for prox_i in prox_graph:
                f.write(str(prox_i))
                f.write('\n')
        prox_graph = []
        # print("i: ", i)

with open('./typeset1.txt', 'w') as f:
    f.write(','.join(list(typeset1)))
with open('./typeset2.txt', 'w') as f:
    f.write(','.join(list(typeset2)))
