# !/usr/bin/python
# -*- coding: utf8 -*-
# D:\Data_D\Downloads\hrEngRi\deprelGraph.py
''' This script parses a conllu file by deprel, stores it as json
pipeline:
1) get corpus in conllu 
2) parse dependencies and store token features with parse_dependency_json in {deprel}/json folder
3) extract features from parsed dependencies and store in {deprel}/extracted folder
4) process collocates as graph
'''
#%% imports
import os, time, json
from collections import Counter
from io import open
import numpy as np
from numpy import nan
from numpy.core.numeric import NaN
import pandas as pd
from pathlib import Path
from conllu import parse_incr, parse
import igraph as ig
import pickle
import json
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet
import networkx as nx
import itertools
import sentiment_functions
#%% Corpus files stored in a dictionary 
connlu_folder = 'conllu' # folder where the conllu files are stored
corpus_files = [
{'file':'engri.vecernji.hr.conllu', 'name':'vecernji', 'id':1},
{'file':'engri.24sata.hr.conllu', 'name':'24sata', 'id':2},
{'file':'engri.novilist.hr.conllu', 'name':'novilist', 'id':3},
{'file':'engri.jutarnji.hr.conllu', 'name':'jutarnji', 'id':4},
{'file':'engri.dnevno.hr.conllu', 'name':'dnevno', 'id':5},
{'file':'engri.direktno.hr.conllu', 'name':'direktno', 'id':6},
{'file':'engri.slobodnadalmacija.hr.conllu', 'name':'slobodnadalmacija', 'id':7},
{'file':'engri.net.hr.conllu', 'name':'net', 'id':8},
{'file':'engri.hrt.hr.conllu', 'name':'hrt', 'id':9},
{'file':'engri.telegram.hr.conllu', 'name':'telegram', 'id':10},
{'file':'engri.index.hr.conllu', 'name':'index', 'id':11},
{'file':'engri.rtl.hr.conllu', 'name':'rtl', 'id':12}
 ] 

# [x['file'] for x in corpus_files] # list all files



#%%
#%% get frequency of lemma within a corpus
def get_freq_json(connlu_folder, conllu_file, freq_folder):
    start = time.time()
    '''opens a conllu file stored in conllu/ folder and counts lemmas and stores in its folder json
    ex:
    {'id': 5, 'form': 'Stormi', 'lemma': 'Stormi', 'upos': 'PROPN', 'xpos': 'Npfsn', 
    'feats': {'Case': 'Nom', 'Gender': 'Fem', 'Number': 'Sing'}, 
    'head': 1, 'deprel': 'conj', 'deps': None, 'misc': {'NER': 'B-PER'}}
    '''
    upos2t= {'VERB':'v', 'NOUN':'n', 'PROPN':'np', 'AUX':'va', 'ADV':'r', 'ADJ':'a', 'PRON':'p', 'NUM':'m', 'DET':'d',
       'X':'x', 'ADP':'s', 'PUNCT':'z', 'PART':'q', 'CCONJ':'c', 'SYM':'y', 'INTJ':'i', 'SCONJ':'cs' }
    lempos_list_all = []
    counter= 0
    data_file = open(connlu_folder+'/'+conllu_file, "r", encoding="utf-8")
    try:
        for tokenlist in parse_incr(data_file):
            # counter+=1
            try:
                for t in tokenlist:
                    try:
                        lemma = t['lemma']
                        pos = upos2t[t['upos']]
                        # print(t['id'], t['form'], t['lemma'], t['upos'], t['deprel'], t['head'])
                        lempos_list_all.append(lemma+'-'+pos) 
                    except:
                        print('Greska u parsiranju tokena.')
                        pass
            except:
                pass
            # if counter == 100000:
            #     break
    except:
        print('Greska u parsiranju tokena.')
        pass
    try:
        lemposS = pd.Series(lempos_list_all)
    except:
        lemposS = pd.Series()
        pass
    lempos_count = lemposS.value_counts() 
    lempos_count.to_json(freq_folder+'/'+conllu_file+'_'+'freq.json')
    print ('get_freq_json for '+conllu_file+' in:'+ str(time.time()- start))
    return lempos_count


def calculate_lema_frequency(freq_folder, save_folder, save_name):
    '''calculates lempos frequency from multiple freq files (.json) 2022 version'''
    dataframes = []
    for file in sorted(os.listdir(freq_folder)):
        print(file)
        data = pd.read_json(freq_folder+'/'+file, orient='index').reset_index().rename(columns={'index':'lempos', 0:'freq'})
        dataframes.append(data)
    frequency = pd.concat(dataframes).groupby(['lempos']).sum().reset_index()
    frequency.to_json(save_folder+'/'+save_name+'_'+'freq.json')

#%% List of all dependencies from https://universaldependencies.org/u/dep/index.html
dependencies = ['nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl', 'vocative',
'expl', 'dislocated', 'advcl', 'advmod', 'discourse', 'aux', 'cop', 'mark',
'nmod', 'appos', 'nummod', 'acl', 'amod', 'det', 'clf', 'case',
'conj', 'cc', 'fixed', 'flat', 'compound', 'list', 'parataxis', 'orphan', 'goeswith', 'reparandum',
'punct', 'root', 'dep']

#%% take conllu file and identify deprel collocates in a sentence
def parse_dependency_json(connlu_folder, conllu_file, deprel, *args):
    start = time.time()
    '''opens a conllu file stored in conllu/ folder and searches for deprel collocates,
    stores tokens with all parsed features in the *deprel/json folder 
    '
    parse_dependency_json('conllu', 'example_data.conllu', 'amod')
    primjer
    json_df= pd.read_json('conj\json\engri.index.hr.conllu_conj_.json')
    example ['head'] ili ['dependency']
    {'id': 5, 'form': 'Stormi', 'lemma': 'Stormi', 'upos': 'PROPN', 'xpos': 'Npfsn', 
    'feats': {'Case': 'Nom', 'Gender': 'Fem', 'Number': 'Sing'}, 
    'head': 1, 'deprel': 'conj', 'deps': None, 'misc': {'NER': 'B-PER'}}
    '''
    deprel_list_all = []
    data_file = open(connlu_folder+'/'+conllu_file, "r", encoding="utf-8")
    counter = -1
    parse_counter= -1
    try:
        for tokenlist in parse_incr(data_file):
            counter += 1
            parse_counter += 1
            tokens=[]
            deprel_t =[]
            # parsing tokens
            try:
                for t in tokenlist:        
                    # print(t['id'], t['form'], t['lemma'], t['upos'], t['deprel'], t['head'])
                    tokens.append(t)
                    if t['deprel'] == deprel:
                        deprel_t.append(t)
            except:
                pass
            for h in deprel_t:
                try:
                    deprel_list_all.append((h,tokens[(int(h['head'])-1)], counter))
                except:
                    pass
    except:
        counter += 1

        # if counter == 500:
        #     break 
    
    # create json directory for storing parsed json files
    try:
        os.makedirs(deprel+'/json/')
    except FileExistsError:
        # if directory already exists
        pass    
    # store as a json
    df = pd.DataFrame().from_dict(deprel_list_all)\
        .rename(columns={0: 'head', 1: 'dependency', 2:'sentence_no'}, inplace=False)
    df.to_json(deprel+'/json/'+conllu_file+'_'+deprel+'_.json')
    print ('parse_dependency_json'+deprel+'/json/'+conllu_file+'_'+deprel+'_.json'+'   time:'+ str(time.time()- start))


def json_to_df_pairs(json_parse_dependency, **kwargs):
    '''reads json with head-dependency of a particular relation and converts it to a df head, dependency, sentence structure'''
    upos2t= {'NOUN':'n', 'PROPN':'n', 
            'ADJ':'a', 
            'VERB':'v',  
            'AUX':'va', 'ADV':'r', 'PRON':'p', 'NUM':'m', 
            'DET':'d', 'X':'x', 'ADP':'s', 'PUNCT':'z', 'PART':'q', 'CCONJ':'c', 
            'SYM':'y', 'INTJ':'i', 'SCONJ':'cs' }
    js_pairs= pd.read_json(json_parse_dependency)
    head= pd.json_normalize(js_pairs['head'])[['id', 'form', 'lemma', 'upos', 'xpos', 'head', 'deprel', 'deps', 'feats.Case', 'feats.Gender']].rename(columns= {'id':'h_id', 'form':'h_form', 'lemma':'h_lemma', 'upos':'h_upos', 'xpos':'h_xpos', 'head':'h_head', 'deprel':'h_deprel', 'deps': 'h_deps', 'feats.Case': 'h_feats.Case', 'feats.Gender': 'h_feats.Gender'})
    head['h_lempos']= head['h_lemma']+'-'+head['h_upos'].map(upos2t)
    dependency = pd.json_normalize(js_pairs['dependency'])[['id', 'form', 'lemma', 'upos', 'xpos', 'head', 'deprel', 'deps', 'feats.Case', 'feats.Gender']].rename(columns= {'id':'s_id', 'form':'s_form', 'lemma':'s_lemma', 'upos':'s_upos', 'xpos':'s_xpos', 'head':'s_head', 'deprel':'s_deprel', 'deps': 's_deps', 'feats.Case': 's_feats.Case', 'feats.Gender': 's_feats.Gender'})
    dependency['s_lempos']= dependency['s_lemma']+'-'+dependency['s_upos'].map(upos2t)
    sentence = js_pairs.sentence_no
    df_pairs = pd.concat([head, dependency, sentence], axis=1)
    df_pairs['weight']= 1
    if kwargs.get('upos_list'):
        df_pairs = df_pairs[(df_pairs['h_upos'].isin(kwargs.get('upos_list'))) | (df_pairs['s_upos'].isin(kwargs.get('upos_list')))]
    if kwargs.get('symetrical_upos')== True:
        df_pairs = df_pairs[(df_pairs['h_upos'] == df_pairs['s_upos'])]
    if kwargs.get('symetrical_pos')== True:
        df_pairs = df_pairs[(df_pairs['h_lempos'].str[-2:] == df_pairs['s_lempos'].str[-2:])]
    return df_pairs 

def df_from_jsons_to_df_pairs(folder_with_json, df_filename, **kwargs):
    df_all = pd.DataFrame()
    for json in sorted(os.listdir(folder_with_json)):
        print(f'Processing {json} file')
        df= json_to_df_pairs(folder_with_json+"/"+json, **kwargs)
        df_all = pd.concat([df_all, df])
        del df
    df_all.to_pickle(df_filename)
    print(f"Files in {folder_with_json} have been writen to {df_filename} as dataframe.")


def extract_deprel_colocations(deprel, deprel_json_file, corpus_id):
    '''
    extract_deprel_colocations from json token - deprel - token
    takes the json_file and parses source, friend keys for creating a graph  
    ex:   extract_deprel_colocations('conj', 'engri.rtl.hr.conllu_conj_.json', 12)
    '''
    start = time.time()
    # 'PROPN':'np'
    upos2t= {'VERB':'v', 'NOUN':'n', 'PROPN':'n', 'AUX':'va', 'ADV':'r', 'ADJ':'a', 'PRON':'p', 'NUM':'m', 'DET':'d',
       'X':'x', 'ADP':'s', 'PUNCT':'z', 'PART':'q', 'CCONJ':'c', 'SYM':'y', 'INTJ':'i', 'SCONJ':'cs' }
    # with open(deprel+'/json/'+deprel_json_file+'_'+deprel+'_.json') as f:
    #     data = json.load(f)
    json_df= pd.read_json(deprel+'/json/'+deprel_json_file)
    json_feature_rows = []
    for _, row in json_df.iterrows():
        try:
            json_feature_rows.append({
                    'lemma_s': row['head']['lemma'], 
                    'upos_s': row['head']['upos'],
                    'lempos_s': row['head']['lemma']+'-'+upos2t[row['head']['upos']],
                    'ner_s': row['head']['misc']['NER'],
                    'lemma_f': row['dependency']['lemma'], 
                    'upos_f' : row['dependency']['upos'], 
                    'lempos_f': row['dependency']['lemma']+'-'+upos2t[row['head']['upos']],
                    'ner_f': row['dependency']['misc']['NER'],
                    'sent': row['sentence_no'], 
                    'corp': corpus_id
                    })
        except:
            pass
    # create extracted folder for storing extracted features  
    try:
        os.makedirs(deprel+'/extracted/')
    except FileExistsError:
        # directory already exists
        pass
    json_feature_rows_df= pd.DataFrame().from_dict(json_feature_rows)
    json_feature_rows_df.to_json(deprel+'/extracted/'+deprel_json_file+'_extract.json')    
    print(deprel_json_file, 'extracted in ', time.time()-start)
    

def extract_multiple_files_deprels(conllu_folder, conllu_files, dependencies):
    '''
    loop all conllu all conllu_files and dependencies -> extract json files
    conllu_folder = 'conllu' #type string
    conllu_files = [x['file'] for x in corpus_files if x['name']=='rtl'] #type list
    dependencies = list of dependencies, ex 'conj' # define deprel that will generate collocates
    read -> primjer_extract = pd.read_json('conj\extracted\engri.dnevno.hr.conllu_extract.json')
    primjer_extract[(primjer_extract['upos_s']==upos) & (primjer_extract['upos_f']== upos) ][['lemma_s', 'lemma_f', 'sent', 'corp']]
    '''
    for conllu_file in conllu_files:
        for deprel in dependencies:
            try: 
                start = time.time()
                print(conllu_file, deprel)
                # 1 parse_dependency into json folder
                parse_dependency_json(conllu_folder, conllu_file, deprel, *args) # parse a conllu file
                # 2 create a df from dict using create_rows_from_json(json_file stored in json folder, corpus_id extracted from corpus_files dict):
                extract_deprel_colocations(deprel ,conllu_file , [x['id'] for x in corpus_files if x['file']==conllu_file][0])
                print(conllu_file, time.time()-start)
            except:
                pass


################ Graph construction 

def construct_graph(extract_folder, extract_json, deprel, save_folder, **kwargs):
    '''
    konstruira graf na temelju extracted_extract_json filea
    ukoliko ima argument pos = 'n' v.... uzima samo taj pos... 
    exclusive_pos = 'n' samo pos koji se odabere ...etc
    mixed_pos = ['n', 'a']
    symetrical_pos True samo koji su isti
    '''
    start=time.time()
    df= pd.read_json(extract_folder+'/'+extract_json)
    try:
        if kwargs.get('exclusive_pos'): #String 
            try:
                df= df[(df['lempos_s'].str.endswith('-'+kwargs.get('exclusive_pos'))) | (df['lempos_f'].str.endswith('-'+kwargs.get('exclusive_pos')))]
            except:
                pass
        if kwargs.get('mixed_pos'): #List
            try:
                df= df[(df['lempos_s'].str.split('-')[1].isin(kwargs.get('mixed_pos'))) | (df['lempos_f'].str.split('-')[1].isin(kwargs.get('mixed_pos')))]
            except:
                pass
        if kwargs.get('symetrical_pos'): #True/False     
            try:
                # df= df.query(('lempos_s')[:-2]==('lempos_f')[:-2])
                # df= df[(df['upos_s']==df['upos_f'])]
                df= df[(df['lempos_s'].str[-2:]==df['lempos_f'].str[-2:])]
            except:
                pass
    except:
        pass
    df_g=pd.DataFrame(df[['lempos_s', 'lempos_f']])
    df_g['weight']= int(1)
    g = ig.Graph.TupleList(df_g.itertuples(index=False), directed=False, weights=True, edge_attrs=None)
    print('Graph '+extract_json+' constructed in: ', time.time()-start)
    try:
        g.write(str(save_folder)+'/'+str(extract_json).replace('.conllu_conj_.json_extract.json', '')+'_'+str(deprel)+'_graph.pkl', format='pickle')
    except:
        print('error saving '+extract_json + 'graph')



def union_graphs(graphs_folder, save_folder, save_name):
    '''uzima grafove, spaja ih, simplificira, dodaje centralnosti, sprema u save_folder pod save_name'''
    g_u= ig.Graph()
    counter = 0
    for graph_file in sorted(os.listdir(graphs_folder)):
        print(os.listdir(graphs_folder))
        # print(graphs_folder+'/'+graph_file)
        g = ig.read(graphs_folder+'/'+graph_file, format='pickle')
        print(len(g.vs), 'prije',g.vs.attributes(), len(g.es))
        if counter == 0:
            g_u = g
        else:
            g_u = g_u.union([g],byname=True)
        counter += 1
    print('Union of graphs', str(sorted(os.listdir(graphs_folder))), g_u.vs.attributes(),len(g_u.vs),g_u.es.attributes(), len(g_u.es))
    g_u = g_u.simplify(combine_edges=dict(weight=sum))
    print('Simplified union of graphs', g_u.vs.attributes(),len(g_u.vs),g_u.es.attributes(), len(g_u.es))
    # g_u.vs["degree"]=g_u.vs.degree()
    # g_u.vs["weighted_degree"] = g.strength(g.vs["name"], weights='weight', mode='ALL')
    # g_u.vs["pagerank"]=g_u.vs.pagerank(directed=False, weights='weight')
    # g_u.vs["sli"]= sli_importance(g_u, igraph=True, normalize=True) #umjesto personalized
    try:
        g_u.write(str(save_folder)+'/'+save_name+'.pkl', format='pickle')
    except:
        print('error saving '+save_name + 'graph')


def construct_multiple_graphs(deprel, corpus_files, save_folder, **kwargs):
    '''
    constructs a graph from a list of extracted json files
    saves the graphs in a *deprel/graphs folder +name+'_'+deprel+'_graph'
    ex:
        deprel= 'conj' .... #string
        corpus_files_names = [x['name'] for x in corpus_files] #list
        **kwargs ex:
            exclusive_pos='n'
    read graph:
        ig.read('conj/graphs/hrt', format='pickle')
    '''
    for file_name in corpus_files: #:
        print('creating '+file_name +' graph')
        #create 1 graph from extracted
        try:
            g = construct_graph(deprel+'/extracted/'+file_name+'_extract.json', **kwargs) 
        except:
            print('error creating graph from '+deprel+'/extracted/'+file_name+'_extract.json')
            pass
        # učitati freq iz freq foldera za korpus koji je učitan
        try:
            with open('freq/'+file_name+'_freq.json') as f:
                freq_file= json.load(f)
            # freqvencija
            freqS = pd.Series(freq_file).reset_index().rename(columns={'index': 'lempos', 0: 'freq'}) 
            # čvorovi raspakiravanje
            gvs_df= pd.Series(g.vs['name']).reset_index().rename(columns={'index': 'index', 0: 'lempos'})
            # mapirati lempos iz freq na lempos u g.vs['name'] 
            out = (gvs_df.merge(freqS, left_on='lempos', right_on='lempos', how='left'))
            g.vs['freq'] = out['freq'] # stavlja freq na čvorove
            # select only the portion of nodes with freqency larger than 1
            g = g.subgraph(g.vs.select(freq_ge= 1))
        except:
            print('error creating '+file_name +' frequency and creating subgraph with more than 1 freq')
                
        # stvara folder    
        try:
            os.makedirs(str(deprel)+'/'+str(save_folder))
        except FileExistsError:
            # directory already exists
            pass 
        try:
            g.write(str(deprel)+'/'+str(save_folder)+'/'+str(file_name)+'_'+str(deprel)+'_graph.pkl', format='pickle')
        except:
            print('error saving '+file_name + 'graph')
        

def graph_union_with_freq_from_graphs(graph_list):
    ''' 
    reads graphs from a list of graph_list 
    example:
    graph_list = list(glob.iglob('conj/graphs/*'))
    creates a union byname, 
    updates freq attribute from sum of all freq_n graphs
    returns a graph_union with node frequencies
    '''
    # 1 create a union from graphs and add all edges to calculate edge weight
    graphs=[]
    for name in graph_list:
        graphs.append(ig.read(name, format='pickle'))
    graph_union = ig.union(graphs, byname=True,)
    graph_union.simplify(combine_edges='sum') # sve ed
    # 2 dodavanje node freqvencija iz različitih korpusa na node atribut freq
    graph_range = list(range(0, graph_union.vcount()))
    lex_freq=[]
    for i in graph_range:
        lex_freq.append(sum(filter(None, [graph_union.vs[i]['freq_'+str(x)] for x in list(range(1,len(graphs)+1))])))
    graph_union.vs['freq']= lex_freq
    # 3 selecting only nodes with freq greater or equal than 1 
    graph_union = graph_union.subgraph(graph_union.vs.select(freq_ge= 1))
    return graph_union

def graph_combine_edges_add_centrality(g):
    '''  
    combine multiple edges of a g graph into unique edges with weight
    get graph centrality measures 
    g = graph with multiple edges #igraph graph
    '''
    g.simplify(combine_edges='sum')
    g.vs["degree"]=g.vs.degree()
    g.vs["pagerank"]=g.vs.pagerank(directed=False, weights='weight')
    g.vs["weighted_degree"] = g.strength(g.vs["name"], weights='weight', mode='ALL')
    # g.vs["sli"]= sli_importance(g, igraph=True, normalize=True) #umjesto personalized
    # g.vs["betweenness"] = g.betweenness() # užasno puno traje da izračuna na cijelom grafu
    print('Number of nodes:', g.vcount(), 'edges:', g.ecount())
    return g



def get_node_frequency_centralities(g_file,freq_folder,freq_name, treshold, save_file):
    '''from a graph pkl file and frequency json file add frequency to nodes and calculate centralities'''
    g= ig.read(g_file, format='pickle')
    try:
        # frequency file
        freqS = pd.read_json(freq_folder+'/'+freq_name+'_freq.json')
        # čvorovi raspakiravanje
        gvs_df= pd.Series(g.vs['name']).reset_index().rename(columns={0: 'lempos'}).drop(columns={'index'})
        # mapirati lempos iz freq na lempos u g.vs['name'] 
        out = (gvs_df.merge(freqS, left_on='lempos', right_on='lempos', how='left'))
        g.vs['freq'] = out['freq'].astype('float32') # stavlja freq na čvorove
        g = g.subgraph(g.vs.select(freq_ge= treshold)) # select only the portion of nodes with freqency larger of equal than 1
    except:
        print('error creating '+freq_name +' frequency and creating subgraph with more than 1 freq')
    g.vs["degree"]=g.vs.degree()
    g.vs["weighted_degree"] =  g.strength(g.vs["name"], weights='weight', mode='ALL') 
    ig.write(g, save_file, format='pickle')
    return g



def graph_add_logDice(g):
    ''' 
    adds logDice measure on the edges as a function of edge weight and node frequencies
    '''
    logDices=[]
    for i in range(0, len(g.es)): 
        e= g.es[i]
        freqR_v1v2 = e['weight']
        v1, v2 = e.source, e.target
        freqN_v1, freqN_v2 = g.vs[v1]['freq'], g.vs[v2]['freq']
        lDice = 14 + np.log2(2*freqR_v1v2/(freqN_v1+freqN_v2))
        logDices.append(lDice)
    g.es['logDice']= logDices
    return g

################ Local analysis functions #################################

def vertex_neighbors(g, lempos, measure):
    ''' 
    find neighbors of a node and its weigted degree
    measure= degree | weighted_degree | pagerank | personalized_pagerank |
    example:
    vertex_neighbors(g, 'kuća-n', 'degree') 
    '''
    neis= g.neighbors(lempos, mode="all")
    return (list(zip(g.vs[neis]['name'], g.vs[neis][measure])))

def find_edge(g, lempos1, lempos2):
    ''' 
    find edge that connects two vertices
    find_edge(g, 'kuća-n', 'lampa-n') 
    '''
    v1= g.vs.find(name=lempos1)
    v2= g.vs.find(name=lempos2)
    try:
        edge_id = g.get_eid(v1, v2, directed=False)
        edge = g.es[edge_id]
    except:
        print('No such edge')
        edge = None
        pass
    return edge

def find_nodes_on_edge(g, edgeID):
    '''
    find source , target on a particular edge
    return source target edege
    '''
    e = g.es[edgeID]
    v1, v2 = e.source, e.target
    return (g.vs[v1],  g.vs[v2], g.es[edgeID])

################## Local data structures 
def first_degree(g, lempos, measure, *args, **kwargs): # đđđđđđđđđđđđđđđđđđ
    '''
    find neighbors of a node and create list with freq
    g -> igraph of dependencies #graph
    lempos -> source lexeme #string 
    measure #string (freq, logDice)
    args -> number of #int
    output #list of dictionaries
    by_index #Boolean searches by node index
    output_index #Boolean returns node index
    ex: first_degree(g,'kuća-n', 'freq', 5, by_index=False, output_index=False)
    '''
    if kwargs.get('by_index'): #True
        v1 = g.vs[int(lempos)]
    else: #False
        v1 = g.vs.find(str(lempos))
    neis= g.neighbors(v1.index, mode="all")
    first_degree=[]
    for v2 in neis:
        try:
            edge= g.get_eid(v1, v2, directed=False)
            if kwargs.get('output_index'): #True
               dic_first = {'source': v1.index, 'friend': g.vs[v2].index, 'freq': g.es[edge]['weight'], 'logDice': g.es[edge]['logDice']} # logDice (14 + np.log2(2*g.es[edge]['weight']/(v1['freq']+g.vs[v2]['freq'])))
            else:
                dic_first = {'source': v1['name'], 'friend': g.vs[v2]['name'], 'freq': g.es[edge]['weight'], 'logDice': g.es[edge]['logDice']} # logDice (14 + np.log2(2*g.es[edge]['weight']/(v1['freq']+g.vs[v2]['freq'])))
            if str(dic_first['logDice'])!='nan':
                first_degree.append(dic_first)
        except:
            pass
    result = sorted(first_degree, key = lambda i: i[measure], reverse= True)[0:args[0]]
    return result

def second_degree(g, lempos, measure, *args, **kwargs):
    '''
    find second degree neighbors of a node and create list with freq
    output #list of dictionaries
    second_degree(g,'kuća-n', 'logDice', 15,5)
    by_index - give index #int as input instead of lempos #str
    output_index - yield index #int instead of lempos #str
    '''
    first_d = first_degree(g, lempos, measure, args[0], by_index=kwargs.get('by_index'), output_index = kwargs.get('output_index'))
    second_dS = []
    for v2 in first_d:
        try:
            second_d = first_degree(g,v2['friend'], measure, args[1], by_index = kwargs.get('output_index'), output_index = kwargs.get('output_index'))
            second_dS = second_dS+ second_d
        except:
            pass
    result = first_d + second_dS
    return result

############# Graph pruning components #################
def pr_es_bet(g_local,set_edgbtw):
    # Prune by edge betweenness
    edgbtw = g_local.edge_betweenness(directed = False, weights='weight')#weights='weight'
    ntile_edge_betweenness = np.percentile(edgbtw, set_edgbtw)
    g_local_pruned_es = g_local.es.select([v for v, b in enumerate(edgbtw) if b >= ntile_edge_betweenness])
    g_local = g_local.subgraph_edges(g_local_pruned_es)
    return g_local

def pr_v_pgrnk(g_local,set_pgrnk):
    # Prune by pagerank
    pgrnk = g_local.pagerank()#weights='weight'
    ntile_pagerank = np.percentile(pgrnk, set_pgrnk)
    g_local_pruned_vs = g_local.vs.select([v for v, b in enumerate(pgrnk) if b >= ntile_pagerank])
    g_local = g_local.subgraph(g_local_pruned_vs)
    return g_local

def pr_v_btwn(g_local,set_btween):
    # Prune by betweenness
    btwn = g_local.betweenness()#weights='weight'
    ntile_betweenness = np.percentile(btwn, set_btween)
    g_local_pruned_vs = g_local.vs.select([v for v, b in enumerate(btwn) if b >= ntile_betweenness])
    g_local = g_local.subgraph(g_local_pruned_vs)
    return g_local

def pr_v_weightdeg(g_local,set_weightdeg):
    # Prune by weighted degree
    weightdeg = g_local.strength(g_local.vs["name"], weights='weight', mode='ALL')
    ntile_weighteddegree = np.percentile(weightdeg, set_weightdeg)
    g_local_pruned_vs = g_local.vs.select([v for v, b in enumerate(weightdeg) if b >= ntile_weighteddegree])
    g_local = g_local.subgraph(g_local_pruned_vs)
    return g_local

def pr_v_sli(g_local,set_sli_limit):
    # Prune by sli
    g_local.vs['sli']= sli_importance(g_local, igraph=True, normalize=True) # count sli
    g_local_pruned_vs = g_local.vs.select(degree_sli = set_sli_limit, sli_le = 100)
    g_local_pruned = g_local.subgraph(g_local_pruned_vs)
    return g_local_pruned



def pr_v_deg(g_local,set_degree_limit):
    # Prune by degree
    g_local.vs['degree']= g_local.degree() # count degree
    g_local_pruned_vs = g_local.vs.select(degree_ge = set_degree_limit, degree_le= 100000)
    g_local_pruned = g_local.subgraph(g_local_pruned_vs)
    return g_local_pruned

# ovo gore pretvoriti u pipeline prunanja

def second_degree_pruned(g, lempos_select, measure, first_n, second_n, set_edgbtw, set_pgrnk, set_btween, set_weightdeg, set_degree_limit, set_resolution):
    '''lempos_select='kuća-n'
    measure= 'logDice' # logDice, freq
    first_n = 15,   second_n = 5
    set_edgbtw #0-100 set the edge betweenness percentile cut
    set_pgrnk  #0-100 set the pagerank percentile cut
    set_btween  #0-100 set the betweenness percentile cut
    set_weightdeg  #0-100 set the weighted degree percentile cut
    set_degree_limit  #0-100  set the degree limit
    set_resolution = 1 #0 less resolution -1 more resolution  ( opcionalno)
    ex: second_degree_pruned(g, 'kuća-n', 'freq', 20, 5, 20, 20, 20, 20, 1, 1)
    '''
    g_local = ig.Graph.TupleList(pd.DataFrame().from_dict(second_degree(g,lempos_select, measure, first_n, second_n)).itertuples(index=False), directed=False, weights=True, edge_attrs=[measure])
    # Prune by edge betweenness
    edgbtw = g_local.edge_betweenness(directed = False, weights='weight')#weights='weight'
    ntile_edge_betweenness = np.percentile(edgbtw, set_edgbtw)
    g_local_pruned_es = g_local.es.select([v for v, b in enumerate(edgbtw) if b >= ntile_edge_betweenness])
    g_local = g_local.subgraph_edges(g_local_pruned_es)
    # Prune by pagerank
    pgrnk = g_local.pagerank()#weights='weight'
    ntile_pagerank = np.percentile(pgrnk, set_pgrnk)
    g_local_pruned_vs = g_local.vs.select([v for v, b in enumerate(pgrnk) if b >= ntile_pagerank])
    g_local = g_local.subgraph(g_local_pruned_vs)
    # Prune by betweenness
    btwn = g_local.betweenness()#weights='weight'
    ntile_betweenness = np.percentile(btwn, set_btween)
    g_local_pruned_vs = g_local.vs.select([v for v, b in enumerate(btwn) if b >= ntile_betweenness])
    g_local = g_local.subgraph(g_local_pruned_vs)
    # Prune by weighted degree
    weightdeg = g_local.strength(g_local.vs["name"], weights='weight', mode='ALL')
    ntile_weighteddegree = np.percentile(weightdeg, set_weightdeg)
    g_local_pruned_vs = g_local.vs.select([v for v, b in enumerate(weightdeg) if b >= ntile_weighteddegree])
    g_local = g_local.subgraph(g_local_pruned_vs)
    # And then prune by betweenness
    g_local.vs['degree']= g_local.degree() # count degree
    g_local_pruned_vs = g_local.vs.select(degree_ge = set_degree_limit, degree_le= 100000)
    g_local_pruned = g_local.subgraph(g_local_pruned_vs)
    # https://igraph.org/c/doc/igraph-Community.html
    print(g_local_pruned.community_leiden( weights='weight', resolution_parameter= set_resolution)) # može se još implementirati resolution_parameter=set_resolution,
    print('Graph diameter: ', g_local_pruned.diameter(), ' vertices: ', g_local_pruned.vcount(), 
    ' edges: ', g_local_pruned.ecount())
    return g_local_pruned

def get_most_similar(graph, direction, number):
    ''' For a (pruned or selected sub) graph calculates:
    the most similar if direction=True
    the most dissimilar if direction=False
    returns similarity number 
    ex: get_most_similar(second_degree_pruned(g, 'kretanje-n', 'freq', 150, 50, 50, 50, 50, 50, 1, 1), True, 10)
    '''
    similar_list= graph.similarity_dice()[0]
    graph.vs['similarity_dice']= similar_list
    return [[v['name'],v['similarity_dice']] for v in sorted(graph.vs, key = lambda i: i['similarity_dice'], reverse = direction)][0:number]

def logDice(freqN_v1, freqN_v2, freqR_v1v2):
    '''
    freqN_v1 - freq node1
    freqN_v2 - freq node2
    freqR_v1v2 - freq edge between 1 2
    ex: logDice(58627.0, 2244.0, 1)
    '''
    lDice = 14 + np.log2(2*freqR_v1v2/(freqN_v1+freqN_v2))
    return lDice

# Shortest path algorithms  https://igraph.org/python/doc/api/igraph._igraph.GraphBase.html#get_shortest_paths

def shortest_by_components(g, v1_name, v2_name):
    '''
    The Graph object in igraph has a method called subcomponent that gives all the nodes 
    that are in the same (weakly connected) component as a given input node. 
    It also has a mode argument. When you set mode to "out", 
    it will give you all the nodes that are reachable from a certain node. 
    When you set mode to "in", it will give you all the nodes from where you can reach a certain node. 
    So, you need the intersection of the set of reachable nodes from your source vertex 
    and the set of nodes that can reach your target vertex.
    '''
    s=set(g.subcomponent(g.vs.find(name=v1_name), mode="out"))
    t=set(g.subcomponent(g.vs.find(name=v2_name), mode="in"))
    return g.vs[s.intersection(t)]['name']

def shortest_paths(g, source, target):
    path = g.get_shortest_paths(source ,to=target, weights='weight', output='vpath')
    result= [g.vs[n] for n in path]
    return path[0]
        
def all_shortest_paths(g, source, target):
    '''
    get_all_shortest_paths(v, to=None, weights=None, mode='out'):
    '''
    path=g.get_all_shortest_paths("virus-n",to="mržnja-n", weights='weight', mode='all')
    for n in path[0]:
        print(g.vs[n]['name'])


################## Local fof
# def fof_initial_dictionary(g,first_n,second_n, write_folder, name):
#     '''
#     For a node in conj graph constructs local fof graph with (n1,n2) 
#     Constructs a list of dictionaries with fof(15,5) for each node in g
#     Writes the dictionary as a python dictionary in write_folder.
#     ex: fof_initial_dictionary(g,first_n,second_n, write_folder, 'fofs_vs_dictionary.py')
        
#     '''
#     start= time.time()
#     fofs_vs_dictionary = []
#     for node in g.vs:
#         try:
#             d= second_degree(g, node['name'], 'logDice', first_n, second_n)
#             if d:
#                 fof_g = ig.Graph.TupleList(pd.DataFrame().from_dict(d).itertuples(index=False), directed=False, weights=True, edge_attrs=None)
#                 fof_g.vs["degree"]=fof_g.vs.degree()
#                 fof_g.vs["pagerank"]=fof_g.vs.pagerank(directed=False, weights='weight')
#                 fof_g.vs["weighted_degree"] = fof_g.strength(fof_g.vs, mode='ALL', weights='weight')
#                 fof_g.vs["betweenness"] = fof_g.betweenness()
#                 fofs_vs_dictionary.append({'source_id':node.index, 'source':node['name'], 'has_fof':True, 'sentic_odv':None, 'vertices': [{'name':x['name'], 'degree': x['degree'], 'weighted_degree':x['weighted_degree'],'pagerank': x['pagerank'], 'sentic_odv':None} for x in list(fof_g.vs)]}) 
#             if not d:
#                 fofs_vs_dictionary.append({'source_id':node.index, 'source':node['name'],'has_fof':False, 'sentic_odv':None})
#         except:
#             pass
#     with open(write_folder+'/fofs_vs_dictionary.py', 'w', encoding='utf-8') as f:
#         print(fofs_vs_dictionary, file=f)
#     print('Done in ', time.time() - start)



################## Sentiment functions
def get_SenticNet_concept_df(lempos, language):
    # '''for a concept list in a language get dataframe of SenticNet values'''
    # this is similar as get_SenticNet_concept(concept, language)
    if language=='deu':
        language='de'
    if language=='en':
        try:
            sn = SenticNet()
            lemma_info = sn.concept(lempos.lower()[0:-2])
        except:
            pass
    else:
        try:
            bsn = BabelSenticNet(language)
            lemma_info = bsn.concept(lempos.lower()[0:-2])
        except:
            pass
    df=pd.DataFrame()
    try: 
        df= df.from_dict(lemma_info, orient='index').transpose() #https://stackoverflow.com/questions/40442014/python-pandas-valueerror-arrays-must-be-all-same-length
        #assign sentics dictionary keys to df columns
        df= df.drop('sentics', 1).assign(**pd.DataFrame(df.sentics.values.tolist())) #https://stackoverflow.com/questions/39640936/parsing-a-dictionary-in-a-pandas-dataframe-cell-into-new-row-cells-new-columns   
        #add label
        df['label']= lempos
    except:
        # if no value, just return label
        # if language=='en':
        #     df=pd.DataFrame(columns=['label', 'polarity_label', 'polarity_value', 'introspection', 'temper', 'attitude', 'sensitivity', 'moodtags', 'semantics'])
        # else:
        #     df=pd.DataFrame(columns=['label', 'polarity_label', 'polarity_value', 'pleasantness', 'attention', 'aptitude', 'sensitivity', 'moodtags', 'semantics'])        
        df= df.append(pd.Series(), ignore_index=True)
        df['label'] = lempos     
    return (df)


def get_SenticNet_c_df(concept_list, language):
    # '''for a concept list in a language get dataframe SenticNet values'''
    concept_values=pd.DataFrame()
    for item in concept_list:
        try:
            concept_values= concept_values.append(get_SenticNet_concept_df(item, language))
        except:
            pass
    return concept_values


def write_odv_sentiment_sentic(g):
    '''
    for a node in graph writes sentiment features on a node
    from an existing SenticNet Dictionary
    the features are prepared for Croatian
    '''
    start = time.time()
    for node in g.vs:
        try:
            sentiment= sentiment_functions.get_SenticNet_concept_df(node['name'], 'hr')
            node['polarity_value']= sentiment['polarity_value'][0]
            node['pleasantness']= sentiment['pleasantness'][0]
            node['attention']= sentiment['attention'][0]
            node['sensitivity']= sentiment['sensitivity'][0]
            node['aptitude']= sentiment['aptitude'][0]
            node['sentic_odv'] = True
        except:
            pass
    
    print('Sentic attributes assigned on', len(g.vs.select(sentic_odv_eq = True)), ' of ', len(g.vs), 'nodes in', time.time()-start, 'seconds')
    return g



def sli_importance(G, **kwargs):
    '''
    The algorithm takes graph G and get the importance of the nodes as a list of floating values
    kwargs: 
        igraph (default False), if True transforms G from igraph to Networkx
        normalize (default True), if False returns non-normalized values from a list of SLI importance values
    '''
    # G introduced in igraph module is transformed into networkx module. If you want to introduce this feature you have to import igraph module
    if kwargs.get('igraph')==True:
        # import igraph
        g=G.to_networkx()
    else:
        g=G
    # print('Graf je tip:', type(g))

    # set edge weight measure to calculate weighted_degree
    edge_measure= 'weight'
    if kwargs.get('weight'):
        edge_measure = kwargs.get('weight')
        
    # print('edge_measure', edge_measure)
    g = nx.DiGraph(g) # mora se prebaciti u DiGraph jer ne radi za Multigraph
    g= g.to_undirected() # pretvara ga u neusmjereni graf
    # print('Nakon konverzije u DiGraph Graf je tip:', type(g))
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis
    cycle = nx.cycle_basis(g) # Will contain: { node_value => sum_of_degrees } 
    # print('cycle:', cycle)
    # print('broj ciklusa:', len(cycle))
    # print('broj čvorova u ciklusima:', [len(x) for x in cycle])

    # weighted_degrees dictionary where keys are gnodes index
    weighted_degrees = g.degree(weight=edge_measure) 
    # print('weighted_degrees:', weighted_degrees)
    nx.set_node_attributes(g, weighted_degrees, "weighted_degree")
    # print('Postavio weighted_degrees na čvorove')

    # Detekcija ciklusa 
    if cycle:   
        # što skuplja degrees i gdje se to kasnije koristi???? 
        # degrees = {}    
        # dict1 se sastoji od komponente edga i komponente broja ciklusa u kojima sudjeluje npr e: 5
        dict1 = {}
        # mi tu detektiramo weighted degree svakog vrha koji sudjeluje na edgu
        for edge in (nx.edges(g)):
            try:
                # print('edge', edge)
                # degrees[edge[0]] = degrees.get(edge[0], 0) + g.degree(edge[1])
                # degrees[edge[1]] = degrees.get(edge[1], 0) + g.degree(edge[0])

                for cicl in cycle:
                    # detektiramo u kojim ciklusima sudjeluje EDGE 
                    # definirano je da na bridu a b je isti weight kao i na b a
                    in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
                    cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
                    in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
                    in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in cycle)
                
                counter_edge=0
                if in_any_cycle(edge, g)==True:
                    # print('edge, in_a_cycle', edge, in_a_cycle)
                    # cicl je par vrhova
                    # prebrojavamo u koliko ciklusa sudjeluje EDGE set(cicl)
                    # counter_edge+=1 
                    for cicl in (cycle):
                        # print('cicl ###########', cicl)
                        c=set(cicl) # set edgeva koji sudjeluju u ciklusu {10,53,12,0}
                        set1=set(edge) # edge koji se tretira kao skup {10,53}
                        is_subset = set1.issubset(c)
                        if is_subset==True:
                            counter_edge+=1 
                    # print(counter_edge)

                else:
                    # print('pass') 
                    pass
                # u dict1 su navedeni bridovi i broj njihovog pojavljivanja u temeljnim cicl
                dict1[edge] = counter_edge
                # u dict1 se p zapisuje na edge a-b i b-a kao ista vrijednost
                # ova linija je bitna za neusmjerene grafove, a ako je usmjeren ona se isključuje
                # treba u slučaju usmjerenih provjeriti i na definiciju ciklusa (cycle)
                dict1[list(itertools.permutations(edge))[1]] = counter_edge
            except:
                pass
        # print('dict1', dict1)
    else:
        pass
        # print('no cycle')

    SLI_importance = []
    # za svaki čvor u grafu 
    for node in nx.nodes(g):
        # print('node ###############',node)
        # print(f'neigh od {node}', list(g.neighbors(node)))  
        sum = 0
        # nodeWeight_node = g.nodes[node]["weighted_degree"]
        #  weighted_degree od noda
        nodeWeight_node = weighted_degrees[node]

        # print('nodeWeight_node', nodeWeight_node)
        # tražimo susjeda za svaki čvor
        for neigh in g.neighbors(node):    
            # print(f'neigh od {neigh}') 
            #  uzimamo težinu svakog brida između čvora i susjeda 
            # edge_weight je težina brida između čvora i susjeda    
            edge_weight= g.get_edge_data(node,neigh)['weight']
            # print('edge_weight', edge_weight)
            
            # to je weighted degree susjeda od čvora u weighted grafu
            nodeWeight_neigh =  weighted_degrees[neigh]
            # print('nodeWeight_neigh', nodeWeight_neigh)
            if not cycle:
                p = 0
            else:
                # p je broj ciklusa u kojima sudjeluje edge
                try:
                    p = dict1[(node, neigh)] # broj ciklusa u kojem sudjeluje edge (node, susjed)
                    # print('p', p)
                except:
                    pass
            u = ((nodeWeight_node + nodeWeight_neigh) - (2 * edge_weight))
            # print('u', u)
            lambd = p+1 # broj ciklusa +1 u kojima sudjeluje edge
            # print('lambd', lambd)
            z = nodeWeight_node / (nodeWeight_node+ nodeWeight_neigh)*edge_weight
            # print('z', z)
            # izračun prema definiciji SLI mjere
            I = u*lambd*z
            # print('I', I)
            sum = sum + I
            # print('sum', sum)
        SLI_importance.append(sum + nodeWeight_node)
        # print('sum + nodeWeight_node', sum + nodeWeight_node)
    SLI_importance= pd.Series(SLI_importance)
    # print('SLI_importance', SLI_importance)
    # print('SLI_importance.sum()', SLI_importance.sum())

    # SLI values non-normalized
    if kwargs.get('normalize') == False:
        SLI_importance_result = SLI_importance.astype('float32')
    # SLI values normalized as default
    else:
        SLI_importance_normalized = SLI_importance/SLI_importance.sum()*100
        SLI_importance_result = SLI_importance_normalized.astype('float32') 
        # print(SLI_importance_result)
    return SLI_importance_result.tolist()




# lista fof grafova u array obliku. ovdje treba staviti i mjeru u argument a ne samo da je logDice
def fof_initial_dictionary_list(g, measure, first_n,second_n, save_file_name):
    ''' 
    after you have a conj dependency graph, and you have write_odv_sentiment_sentic, run this function to get array 
        arguments
    g - graph 
    measure - weight measure (freq, logDice, etc.)
    first_n
    second_n
    save_file_name
        creates:
    source lista: source_id, fof_n, sentic_odv
    vertexes lista: target_id, degree, weighted_degree, pagerank, betweeness, sentic_odv
    last element: (odv true + false in vertices) / len(vertices) 
    Source: 
            source_id #int
            len(fof) #int
            sentic_odv # bool + None
        Vertices #dictionary list
            vertex_id #int
            degree #int
            weighted_degree #float
            pagerank #float
            betweeneess # float
            sentic_odv #Bool + None
        Certainty (broj odv/broj svih u Vertices)
    '''
    start= time.time()
    fofs_vs = []
    for node in g.vs():
        try:
            d= second_degree(g, node.index, measure, first_n, second_n, by_index = True, output_index = True)
            if d:
                fof_g = ig.Graph.TupleList(pd.DataFrame().from_dict(d).itertuples(index=False), directed=False, weights=True, edge_attrs=None)
                fof_g.vs["sli"] = sli_importance(fof_g, igraph=True, normalize=True)
                fof_g.vs["degree"]=fof_g.vs.degree()  
                fof_g.vs["weighted_degree"] = fof_g.strength(fof_g.vs, mode='ALL', weights='weight')
                fof_g.vs["pagerank"]=fof_g.vs.pagerank(directed=False, weights='weight')
                fof_g.vs["betweenness"] = fof_g.betweenness()
                
                # source lista: 'source_id', 'fof_n', 'sentic_odv',
                source = [int(node.index), len(fof_g.vs), node['sentic_odv']]
                
                # vertex lista: id, degree, weighted_degree, pagerank, sentic_odv
                vertices = [[int(x['name']), float(x['sli']), int(x['degree']), float(x['weighted_degree']), float(x['pagerank']), float(x['betweenness']), g.vs[x['name']]['sentic_odv']] for x in fof_g.vs]    # ! betweenness je naknadno ugrađen            
                fofs_vs.append(source + vertices + [0])
            # if there is no fof graph              
            if not d:
                source= [int(node.index), 0, 0]
                fofs_vs.append(source + [None] + [0])
        except:
            pass
    arr = np.array(fofs_vs,dtype=object)
    # calculate the percentage of the True values in the verexes
    for item in range(0, len(arr)): # node
        try:
            c= Counter([x[6] for x in arr[item][3:3+arr[item][1]]])
            arr[item][-1] = round((c[True]+c[False])/arr[item][1], 2)        
        except:
            arr[item][-1] = 0
    np.save(save_file_name, arr, allow_pickle=True)
    print('Array stored in ', save_file_name, ' Finished in ', time.time() - start)
    # ~ 2500 seconds
    return arr







def change_odv_status_subsequent(arr, nodes_for_change, status_change):
    '''
    goes through the arr and changes the status of the node and node's vertices in fof
    '''
    # change source node odv_value to status_change
    for node in nodes_for_change: # node
        arr[node][2] = status_change    
    
    # change the odv_value of node's vertices to status_change
    for item in range(0, len(arr)):
        for x in arr[item][3:3+arr[item][1]]:
            if x[0] in nodes_for_change:
                x[5] = status_change
                
                #update the fof_p values for list of changed vertices
                try:
                    c= Counter([x[5] for x in arr[item][3:3+arr[item][1]]])
                    arr[item][-1] = round((c[True]+c[False])/arr[item][1], 2)        
                except:
                    arr[item][-1] = 0
        
    # print('Changed status for ', len(nodes_for_change), 'items in array')           
    return arr

def get_fill_steps(arr, file_dump_name):
    '''
    takes the array, finds the nodes with odv None , collects the node and fof_p 
    selects nodes with max fof_p, changes odv to False for the node and vertices
    '''

    fill_list=[] #lista tuplova : vrijednost fof_p, lista_čvorova_za_puniti
    maximum = 1
    while maximum > 0: #Traži max dok ne dođe do 0
        # skuplja listu tupla vrijednosti  vrijednosti za čvorove koji nisu None
        lst = [(arr[n][0], arr[n][-1]) for n in range(0, len(arr)) if arr[n][2] == None]     
        print('Number of nodes with None: ', len(lst))
        #pronađe maximalnu vrijednost
        maximum = max([tup[1] for tup in lst])     
        # traži čvorove sa maximalnom vrijednosti iz liste tuplova
        nodes_for_change = []
        [nodes_for_change.append(tup[0]) for tup in lst if tup[1] == maximum]
        print('Nodes to change the status at max', maximum, ' - ', len(nodes_for_change))
        fill_list.append((maximum, nodes_for_change))
        # change nodes
        arr = change_odv_status_subsequent(arr, nodes_for_change, False)
        #print('nodes filled in: ', time.time()-start)

    #%% spremiti listu kao pickle
    with open(file_dump_name, "wb") as fp:   #Pickling
        pickle.dump(fill_list, fp)
    return(fill_list)



def get_fill_steps_and_final_graph(g, arr, file_dump_name):
    '''
    takes the array, finds the nodes with odv None , collects the node and fof_p 
    selects nodes with max fof_p, changes odv to False for the node and vertices
    '''
    fill_list=[] #lista tuplova : vrijednost fof_p, lista_čvorova_za_puniti
    maximum = 1
    while maximum > 0: #Traži max dok ne dođe do 0
        # skuplja listu tupla vrijednosti  vrijednosti za čvorove koji nisu None
        lst = [(arr[n][0], arr[n][-1]) for n in range(0, len(arr)) if arr[n][2] == None]     
        print('Number of nodes with None: ', len(lst))
        #pronađe maximalnu vrijednost
        maximum = max([tup[1] for tup in lst])     
        # traži čvorove sa maximalnom vrijednosti iz liste tuplova
        nodes_for_fill = []
        [nodes_for_fill.append(tup[0]) for tup in lst if tup[1] == maximum]
        print('Nodes to change the status at max', maximum, ' - ', len(nodes_for_fill))
        fill_list.append((maximum, nodes_for_fill))
        # Sentiment calculation section
        sentiment_list = [] # list with all the sentiment calculations
        for node_for_fill in nodes_for_fill:
            # change the sentiment values in the graph
            centrality_sli=[]
            centrality_degrees=[] 
            centrality_w_degrees=[]
            centrality_pageranks=[]
            centrality_betweenness=[]
            polarity_values = []
            pleasantnesss  = []
            attentions = []
            sensitivitys = []
            aptitudes  = []      
            # Find node_for_fill vertices with ODV True or False (only the filled ones)
            for fof_vs in [x for x in arr[node_for_fill][3:3+arr[node_for_fill][1]] if x[6]==True or x[6]==False]:        
                # get fof_vs's centrality values from array
                centrality_sli.append(fof_vs[1])
                centrality_degrees.append(fof_vs[2])
                centrality_w_degrees.append(fof_vs[3])
                centrality_pageranks.append(fof_vs[4])
                centrality_betweenness.append(fof_vs[5])
                # za svaku sentmeasure iz liste pronađi vrijednosti verteksa
                # for sentmeasure in ['polarity_value', 'pleasantness', 'attention', 'sensitivity', 'aptitude']:
                # uzmi broj čvora i pogledaj u g.vs koji mu je sentiment value
                polarity_values.append(g.vs[fof_vs][0]['polarity_value'])
                pleasantnesss.append(g.vs[fof_vs][0]['pleasantness'])
                attentions.append(g.vs[fof_vs][0]['attention'])
                sensitivitys.append(g.vs[fof_vs][0]['sensitivity'])
                aptitudes.append(g.vs[fof_vs][0]['aptitude'])
            # izračunaj srednjicu na temelju sentmeasure i centralnosti 
            measure_counter=0
            for centrality in [centrality_sli, centrality_degrees, centrality_w_degrees, centrality_pageranks, centrality_betweenness]:
                measure_counter=measure_counter+1
                try:
                    polarity_value = srednjica(polarity_values, centrality)
                    pleasantness = srednjica(pleasantnesss, centrality)
                    attention = srednjica(attentions, centrality)
                    sensitivity = srednjica(sensitivitys, centrality)
                    aptitude = srednjica(aptitudes, centrality)
                    # upiši vrijednost sentimenta izračunatog po centrality u listu
                    sentiment_list.append((node_for_fill, polarity_value, pleasantness, attention, sensitivity, aptitude, maximum, measure_counter))
                except:
                    sentiment_list.append((node_for_fill, None, None, None,None,None, maximum))
                    pass
            # upiši vrijednost sentimenta izračunatog po centrality u graph
            # vrijednost će biti zadnji u listi centrality
            try:
                g.vs[node_for_fill]['polarity_value']= polarity_value
                g.vs[node_for_fill]['pleasantness']= pleasantness
                g.vs[node_for_fill]['attention']= attention
                g.vs[node_for_fill]['sensitivity']= sensitivity
                g.vs[node_for_fill]['aptitudes']= aptitude
                print(node_for_fill, polarity_value)
            except:
                print(node_for_fill, 'no polarity_value')
                pass
            # promijeni sentic_odv vrijednost u False
            g.vs[node_for_fill]['sentic_odv'] = False
            # zabilježi sigurnost izračunate adv vrijednosti kao omjer napunjenih / svih čvorova fof
            g.vs[node_for_fill]['adv_cert']= maximum              
        
            # change odv_value status in array
            arr = change_odv_status_subsequent(arr, [node_for_fill], False)
        

    #%% spremiti liste kao pickle
    np.save(file_dump_name+'_fill_list.pkl', fill_list, allow_pickle=True)
    np.save(file_dump_name+'_sentiment_list.pkl', sentiment_list, allow_pickle=True)
    ig.write(g, file_dump_name+'_graph.pkl', format='pickle')
    return(g, fill_list, sentiment_list)




def get_fill_steps_and_final_graph_naive(g, arr, file_dump_name):
    '''
    takes the array, finds the nodes with odv None , collects the node and fof_p 
    calculates the sentic_odv for the nodes with odv None
    '''
    fill_list=[] #lista tuplova : vrijednost fof_p, lista_čvorova_za_puniti

    # skuplja listu tupla vrijednosti  vrijednosti za čvorove koji nisu None
    lst = [(arr[n][0], arr[n][-1]) for n in range(0, len(arr)) if arr[n][2] == None]     
    print('Number of nodes with None: ', len(lst))
    nodes_for_fill = [tup[0] for tup in lst]
    print('Nodes to change the status', len(nodes_for_fill))
    # Sentiment calculation section
    sentiment_list = [] # list with all the sentiment calculations
    for node_for_fill in nodes_for_fill:
        # calculate the certainty 
        maximum = len([x for x in arr[node_for_fill][3:(3+arr[node_for_fill][1])] if x[6]==True or x[6]==False]) / arr[node_for_fill][1]
        fill_list.append((maximum, node_for_fill))

        # change the sentiment values in the graph
        centrality_sli=[]
        centrality_degrees=[] # maknuli smo centralnost po degree i zamijenili sa sli
        centrality_w_degrees=[]
        centrality_pageranks=[]
        centrality_betweenness=[]
        polarity_values = []
        pleasantnesss  = []
        attentions = []
        sensitivitys = []
        aptitudes  = []      
        # Find node_for_fill vertices with ODV True or False (only the filled ones)
        for fof_vs in [x for x in arr[node_for_fill][3:3+arr[node_for_fill][1]] if x[6]==True or x[6]==False]:        
            # get fof_vs's centrality values from array
            centrality_sli.append(fof_vs[1])
            centrality_degrees.append(fof_vs[2])
            centrality_w_degrees.append(fof_vs[3])
            centrality_pageranks.append(fof_vs[4])
            centrality_betweenness.append(fof_vs[5])
            # za svaku sentmeasure iz liste pronađi vrijednosti verteksa
            # for sentmeasure in ['polarity_value', 'pleasantness', 'attention', 'sensitivity', 'aptitude']:
            # uzmi broj čvora i pogledaj u g.vs koji mu je sentiment value
            polarity_values.append(g.vs[fof_vs][0]['polarity_value'])
            pleasantnesss.append(g.vs[fof_vs][0]['pleasantness'])
            attentions.append(g.vs[fof_vs][0]['attention'])
            sensitivitys.append(g.vs[fof_vs][0]['sensitivity'])
            aptitudes.append(g.vs[fof_vs][0]['aptitude'])
            # izračunaj srednjicu na temelju sentmeasure i centralnosti 
        measure_counter=0
        for centrality in [centrality_sli, centrality_degrees, centrality_w_degrees, centrality_pageranks, centrality_betweenness]:
            measure_counter=measure_counter+1
            try:
                polarity_value = srednjica(polarity_values, centrality)
                pleasantness = srednjica(pleasantnesss, centrality)
                attention = srednjica(attentions, centrality)
                sensitivity = srednjica(sensitivitys, centrality)
                aptitude = srednjica(aptitudes, centrality)
                # upiši vrijednost sentimenta izračunatog po centrality_w_degrees na graf
                sentiment_list.append((node_for_fill, polarity_value, pleasantness, attention, sensitivity, aptitude, maximum, measure_counter))
            except:
                sentiment_list.append((node_for_fill, None, None, None,None,None, maximum))
                pass
        try:
            g.vs[node_for_fill]['polarity_value']= polarity_value
            g.vs[node_for_fill]['pleasantness']= pleasantness
            g.vs[node_for_fill]['attention']= attention
            g.vs[node_for_fill]['sensitivity']= sensitivity
            g.vs[node_for_fill]['aptitudes']= aptitude
            print(node_for_fill, polarity_value)
        except:
            print(node_for_fill, 'no polarity_value')
            pass
        # promijeni sentic_odv vrijednost u False
        g.vs[node_for_fill]['sentic_odv'] = False
        # zabilježi sigurnost izračunate adv vrijednosti kao omjer napunjenih / svih čvorova fof
        g.vs[node_for_fill]['adv_cert']= maximum              
    
        # change odv_value status in array
        arr = change_odv_status_subsequent(arr, [node_for_fill], False)
    # spremiti liste kao pickle
    # np.save(file_dump_name+'_fill_list_naive.pkl', fill_list, allow_pickle=True)
    # np.save(file_dump_name+'_sentiment_list_naive.pkl', sentiment_list, allow_pickle=True)
    ig.Graph.write_pickle(g, file_dump_name+'_graph_naive.pkl')
    return(g, fill_list, sentiment_list)




def fill_analysis_values(fill_steps):
    percentages=[]
    counter=0
    for x in fill_steps:
        counter += 1
        percentages.append((counter, x[0]))
    percentages 
    p= pd.DataFrame({'step':[x[0] for x in percentages], 'value':[x[1] for x in percentages]})  
    res1 = p.plot.scatter(x='step', y='value', figsize=[20,5])
    res2 = p.describe()['value']
    return (res1,res2) 

def fill_analysis_node_numbers(fill_steps):    
    numbers=[]
    counter=0
    for x in fill_steps:
        counter += 1
        numbers.append((counter, (len(x[1]))))
    numbers 
    nn = pd.DataFrame({'step':[x[0] for x in numbers], 'node numbers':[x[1] for x in numbers]})  
    res1 = nn.plot.scatter(x='step', y='node numbers', figsize=[20,5])
    res2 = nn.describe()['node numbers']
    return (res1,res2)


def srednjica(val_list, node_importance_values_list):
    '''
    for a source node with the similar neighbours in the undirected graph, 
    calculates the value projected by the neighbours  
    val_list - sentiment values of nodes #list 
    node_importance values - node centrality measures #list 
    ex: srednjica([3,4,20],[4,1,1]) yields 6.0
    '''
    brojnik=[]
    nazivnik= node_importance_values_list
    for item1 in zip(val_list,node_importance_values_list):
        brojnik.append(item1[0]*item1[1])
    return (sum(brojnik)/sum(nazivnik))



def fill_values(g, arr, fill_steps, file_dump_name):
    counter=0
    fill_tuples = []
    for step in enumerate(fill_steps): 
        counter=counter+1
        print('Percent steps:', round(counter/len(fill_steps)*100,2), 'Nodes in current step:', len(step[1][1])) 
        certainty= step[1][0]
        # za svaki node koji je za puniti sa vrijednosti sentimenta u koraku
        for node in step[1][1]:
            centrality_sli=[]
            centrality_w_degrees=[]
            centrality_pageranks=[]
            centrality_betweenness=[]
            polarity_values = []
            pleasantnesss  = []
            attentions = []
            sensitivitys = []
            aptitudes  = []      
            # pronađi njegove vertekse sa ODV True ili False (oni koji su napunjeni)
            for fof_vs in [x for x in arr[node][3:3+arr[node][1]] if x[5]==True or x[5]==False]:        
                # uzmi centrality vrijednosti iz arraya
                centrality_sli.append(fof_vs[1])
                centrality_w_degrees.append(fof_vs[2])
                centrality_pageranks.append(fof_vs[3])
                centrality_betweenness.append(fof_vs[4])
                # za svaku sentmeasure iz liste pronađi vrijednosti verteksa
                # for sentmeasure in ['polarity_value', 'pleasantness', 'attention', 'sensitivity', 'aptitude']:
                    # uzmi broj čvora i pogledaj u g.vs koji mu je sentiment value
                polarity_values.append(g.vs[fof_vs][0]['polarity_value'])
                pleasantnesss.append(g.vs[fof_vs][0]['pleasantness'])
                attentions.append(g.vs[fof_vs][0]['attention'])
                sensitivitys.append(g.vs[fof_vs][0]['sensitivity'])
                aptitudes.append(g.vs[fof_vs][0]['aptitude'])
                # izračunaj srednjicu na temelju sentmeasure i centralnosti 
            measure_counter=0
            for centrality in [centrality_sli, centrality_w_degrees, centrality_pageranks, centrality_betweenness]:
                measure_counter=measure_counter+1
                try:
                    polarity_value = srednjica(polarity_values, centrality)
                    pleasantness = srednjica(pleasantnesss, centrality)
                    attention = srednjica(attentions, centrality)
                    sensitivity = srednjica(sensitivitys, centrality)
                    aptitude = srednjica(aptitudes, centrality)
                    # upiši vrijednost sentimenta izračunatog po centrality_w_degrees na graf
                    fill_tuples.append((node, polarity_value, pleasantness, attention, sensitivity, aptitude, certainty, measure_counter))
                except:
                    fill_tuples.append((node, None, None, None,None,None, certainty))
                    pass
            try:
                g.vs[node]['polarity_value']= polarity_value
                g.vs[node]['pleasantness']= pleasantness
                g.vs[node]['attention']= attention
                g.vs[node]['sensitivity']= sensitivity
                g.vs[node]['aptitudes']= aptitude
            except:
                pass
            # promijeni sentic_odv vrijednost u False
            g.vs[node]['sentic_odv']= False
            # zabilježi sigurnost izračunate adv vrijednosti kao omjer napunjenih / svih čvorova fof
            g.vs[node]['adv_cert']= certainty              
            #change the odv_value to False in array
            change_odv_status_subsequent(arr, [node], False)
    # save results: fill_tuples with calculated values and graph
    with open(file_dump_name+'_tuples.pkl', "wb") as fp:   #Pickling
        pickle.dump(fill_tuples, fp)
    ig.Graph.write_pickle(g, file_dump_name+'_graph.pkl')
    print('Done')
    return g





# senticnet srednjica
def get_sentic_srednjica_df(SenticNet_c_df, node_importance_values_list, node_importance_measure):
#     # for a dataframe of  sentic values, nodes in a graph, measure of node importance get dataframe of sentic_values 
#     # take list of concepts, values, and its node importance and return the middle value
#     # SenticNet_c_df je df sa lemama za koju se traži vrijednost, 
#     # node importance_values_list je lista vrijednosti važnosti čvora, 
#     # node_importanc_measure je naziv mjere značaja čvora u grafu
    SenticNet_c_df=SenticNet_c_df.dropna()
    sentic_df =pd.DataFrame(index=[node_importance_measure])
    for key in SenticNet_c_df.keys(): # for every column create srednjica
        if not key in ['label', 'moodtags', 'semantics', 'polarity_label']:
            sentic_df[key]= srednjica([float(x) for x in SenticNet_c_df[key]], node_importance_values_list )
    return sentic_df




def get_graph_sentic_values(SenticNet_c_df, graph_df):
    # for a dataframe of  sentic values, nodes in a graph get dataframe of sentic_values according to a listOf node_importance_measures
    node_importance_measures=['pagerank', 'sli', 'weighted_degree', 'betweenness']
    graph_sentic_values = pd.DataFrame()
    # try:
    for node_importance_measure in node_importance_measures: 
        # get node importance from graph_DataFrame for all labels containing some value in sentic_concepts_df
        node_importance_values_list = graph_df[graph_df["label"].isin([str(x) for x in SenticNet_c_df[SenticNet_c_df['polarity_value'].notna()]['label']])][node_importance_measure].astype('float').tolist()
        graph_sentic_values= graph_sentic_values.append(get_sentic_srednjica_df(SenticNet_c_df, node_importance_values_list, node_importance_measure))
    # except:
    #     pass
    return (graph_sentic_values)



def calculate_sentic_values(g, nodes, measure, first_n, second_n, language):
    # for a lemma in df_senticnet get sourceValuePropagation 2- napraviti izračun po metodi - sourceValuePropagation: val(sVP)
    # pos = -n, -v, -j
    # start=time.time() # measure time
    # measure : freq | logDice
    sentic_calculated_values=pd.DataFrame() # create a DataFrame
    for node in nodes: 
        print(fof_g)
        d= second_degree(g, g.vs[node]['name'], measure, first_n, second_n)
        fof_g = ig.Graph.TupleList(pd.DataFrame().from_dict(d).itertuples(index=False), directed=False, weights=True, edge_attrs=None)
        fof_g.vs["sli"]= sli_importance(fof_g, igraph=True, normalize=True) 
        fof_g.vs["degree"]=fof_g.vs.degree()
        fof_g.vs["pagerank"]=fof_g.vs.pagerank(directed=False, weights='weight')
        fof_g.vs["weighted_degree"] = fof_g.strength(fof_g.vs, mode='ALL', weights='weight')
        fof_g.vs["betweenness"] = fof_g.betweenness()
        fof_snp_graph_df = pd.DataFrame({attr: fof_g.vs[attr] for attr in fof_g.vertex_attributes()})
        # get sentic values for lexical nodes in FoF
        sentic_snp = get_SenticNet_c_df(fof_g.vs["name"], language)
        # get sentic value for a lemma in dictionary based on the  'sentic_snp '+lemma, sentic_snp
        sentic_snp_value = get_graph_sentic_values(sentic_snp, fof_snp_graph_df)
        sentic_snp_value['label'] = node.vs['name']
        sentic_calculated_values = sentic_calculated_values.append(sentic_snp_value)
        
    return sentic_calculated_values