#!/usr/bin/python
# -*- coding: utf8 -*-
# 2022-03 version
#%% D:\Data_D\Downloads\hrEngRi\bura_engri_2.py
''' This script parses a conllu file by deprel, stores it as json
pipeline:
1) get corpus in conllu 
2) parse dependencies and store token features with parse_dependency_json in {deprel}/json folder
3) extract features from parsed dependencies and store in {deprel}/extracted folder
4) process collocates as graph
'''
#%%
import os, time, json
from collections import Counter
from io import open
# import kafka
import numpy as np
import pandas as pd
import glob     
from conllu import parse_incr, parse
import networkx as nx
import igraph as ig
import itertools
import deprelGraph as dG 
print('libraries and modules loaded')
# import classla
#%% Priprema ulaznih podataka
# Conllu files  
conllu_folder = 'conllu/hrwac_corr' 
corpus_files = sorted(os.listdir(conllu_folder))# folder where the conllu files are stored
# corpus_files = dG.corpus_files
# corpus_files

#%% List of dependencies (from https://universaldependencies.org/u/dep/index.html)
dependencies = dG.dependencies
print(dependencies)
# Definiranje jedne dependencije 
deprel= 'conj'


#%%######## Parsiranje: uzima parsirani tekst i ekstrahira pojedinačne instance deprel kolokacija
# 1 activate parse_dependency_json and get deprel collocates stored as json files in a folder defined by function 
# for corpus_file in corpus_files:
#     dG.parse_dependency_json(conllu_folder, corpus_file, 'conj')
#%%
# Example of a json file  read json_df
json_df= pd.read_json('conj\json\hrwac_parsed_nonstandard_1-56_0001.conllu_conj_.json')
# json_df

#%%

### 2022 -03 dodatak pretvaranje u DataFrame
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

# spoji datasetove

#kako napraviti g od dataseta
# df= json_to_df_pairs('conj/novi/hrwac_01.conllu_conj_.json', symetrical_pos=True)
#%%
df= json_to_df_pairs('conj\json\hrwac_parsed_nonstandard_1-56_0003.conllu_conj_.json', upos_list=['NOUN'], symetrical_upos=True, symetrical_pos=False)

# save df to pickle
df.to_pickle('conj\json\hrwac_parsed_nonstandard_1-56_0003.conllu_conj_.pkl')

#%%
# summarize df by weight
df.groupby(['h_lempos', 's_lempos']).sum().sort_values(by='weight', ascending=False).head(20)
#%%
# create df with sum of weights
df= df.groupby(['h_lempos', 's_lempos']).sum().reset_index()

#%%
# create nx graph
G= nx.from_pandas_edgelist(df, 'h_lempos', 's_lempos', 'weight', create_using=nx.DiGraph)
#%%
G= ig.Graph.DataFrame(df[['h_lempos','s_lempos','weight']])

G= G.simplify(combine_edges=dict(weight=sum))


len(G.vs()), len(G.es()), G.es[2], G.vs[4], [x['name'] for x in G.vs()]



#%% 2 create extract_deprel_colocations: ekstrahira deprel kolokacije
dG.extract_deprel_colocations('conj', [x+'_'+deprel+'_.json' for x in corpus_files][0], 'hrwac')



#%%
json_df= pd.read_json(str(deprel)+'/json/'+str([x+'_'+deprel+'_.json' for x in corpus_files][0]))
print(json_df)
# ~ min per 100 mb
# 2.1. read extracted json_df
# primjer_extract = pd.read_json('conj\extracted\engri.dnevno.hr.conllu_extract.json')
#%% pretraži po upos
# upos='VERB'
# primjer_extract[(primjer_extract['upos_s']==upos) & (primjer_extract['upos_f']== upos) ][['lemma_s', 'lemma_f', 'sent', 'corp']]

#%%######### Graph construction using IGRAPH 
# construct graphs from a folder
extract_folder= 'conj/extracted'
graphs_folder= 'conj/graphs_hrwac'

#%%
# for extract_json in sorted(os.listdir(extract_folder)): 
#     dG.construct_graph(extract_folder, extract_json,'conj', graphs_folder , symetrical_pos = True)


#%% budući da graph union ne radi kako treba, spojit ćemo sve json filove, i onda ih transformirati u multigraf, zatim simplificirati u graph
#%%
import pandas as pd
df_all=pd.DataFrame()
for extract_json in sorted(os.listdir(extract_folder)): 
    print (extract_json)
    df= pd.read_json(extract_folder+'/'+extract_json)
    df= df[(df['lempos_s'].str[-2:]==df['lempos_f'].str[-2:])]
    df_g=pd.DataFrame(df[['lempos_s', 'lempos_f']])
    print(df_g.head())
    df_all= pd.concat([df_all, df_g], ignore_index=True, axis=0)

df_all['weight']= 1
df_all.to_pickle('df_conj_all_symetric.pkl')
#%%
df_all= pd.read_pickle('df_conj_all_symetric.pkl')
#%% za izračunati sve pojavnice koje su u mreži
df_all_pojavnice = pd.DataFrame(pd.concat([df_all.lempos_s, df_all.lempos_f]).unique().tolist(), columns=['lempos'])

#%% učitati sve pojavnice iz korpusa
sve_pojavnice = pd.read_json('freq\hrwac\hrwac_lempos_freq.json')

#%% vidjeti koje su sve pojavnice u grafu
merged = sve_pojavnice.merge(df_all_pojavnice, how= 'outer', left_on='lempos', right_on='lempos' )
#%% vidjeti koje su sve pojavnice izvan 
out_merged = sve_pojavnice.merge(df_all_pojavnice, how= 'outer', left_on='lempos', right_on='lempos' )
#%%
len(sve_pojavnice[(sve_pojavnice['lempos'].str.endswith('-n')) &  (sve_pojavnice['freq']>5)]) # 117517
#%%
g = ig.Graph.TupleList(df_all.itertuples(index=False), directed=False, weights=True, edge_attrs=None)
print('Graph_all constructed')
g.write('g_all_graph.pkl', format='pickle')

#%% proba
pd.read_csv('conj\extracted\proba_veze.csv', sep=';').to_json('conj\extracted\proba_veze.json')
#%%
dG.construct_graph('conj/extracted/', 'proba_veze.json','conj', 'conj/extracted/' , symetrical_pos = True)
pg= ig.load('conj\extracted\proba_veze.json_conj_graph.pkl', format='pickle')
print(len(pg.vs), len(pg.es))
pgn= pg.to_networkx()
labels = nx.get_node_attributes(pgn, 'name') 
nx.draw(pgn,labels=labels,node_size=100)
#%%%%
pg2= ig.load('conj\extracted\proba\proba_veze2.json_conj_graph.pkl', format='pickle')

#%%
dG.union_graphs('conj/extracted/proba/', 'conj\extracted\proba/','union_conj_proba')
#%%
ug= ig.load('conj\extracted\proba/union_conj_proba.pkl', format='pickle') 
#%% oooooooovooooooo radi probleeeeeemememememememe
graphs_folder= 'conj/extracted/proba/'
g_u= ig.Graph()
counter = 0
for graph_file in sorted(os.listdir('conj/extracted/proba/')):
    print('counter',counter)
    print('file: ', graph_file)
    # print(graphs_folder+'/'+graph_file)
    g = ig.read(graphs_folder+'/'+graph_file, format='pickle')
    print(len(g.vs), 'prije',g.vs.attributes(), len(g.es))
    if counter == 0:
        g_u = g
    if counter>0:
        g_u = g_u.union(g,byname=True)
    print('broj veza nakon unije', len(g_u.es))
    counter += 1
print('Union of graphs', str(sorted(os.listdir(graphs_folder))), g_u.vs.attributes(),len(g_u.vs),g_u.es.attributes(), len(g_u.es))
for e in g_u.es:
    print(e)
#%%
g_u = g_u.simplify(combine_edges=dict(weight=sum))
print('Simplified union of graphs', g_u.vs.attributes(),len(g_u.vs),g_u.es.attributes(), len(g_u.es))



#%%
ug= ig.load('conj/extracted/proba/union_conj_proba.pkl', format='pickle')
ugn= ug.to_networkx()
labels = nx.get_node_attributes(ugn, 'name') 
nx.draw(ugn,labels=labels,node_size=100)

#%% učitaj sve grafove, pojednostavi veze (Multigraph > Simple Graph), zbroji weight na vezama

dG.union_graphs(graphs_folder, graphs_folder,'hrwac_union_conj_proba')
#%% pospojiti freq iz fileova  
freq_folder= 'freq/hrwac'
freq_name = 'hrwac_lempos'

dG.calculate_lema_frequency(freq_folder, freq_folder, freq_name)
#%% na probi zaključeno da ova funkcija dobro radi
# dG.calculate_lema_frequency('freq/hrwac/proba', 'freq/hrwac/proba', 'proba.json')

#%% staviti freq na graph hrwac
# učitati freq iz freq foldera za korpus koji je učitan


dG.get_node_frequency_centralities(graph_file ,freq_folder,freq_name, 1, 'conj\graphs_hrwac\proba')
    

#%%
# gx= dG.graph_combine_edges_add_centrality(g_u)

#%%# construct graph from multiple deprel extracted json_df
names_list= [x['name'] for x in corpus_files]
g= dG.construct_multiple_graphs('conj', names_list )
#%% after construction of individual graphs unite graphs 
graph_list=list(glob.iglob('conj/graphs_hrwac/*')) #ili nešto drugo
print(graph_list)
graphs=[]
for name in graph_list:  
    graphs.append(ig.read(name, format='pickle'))
graphs
#%%
gx= dG.graph_union_with_freq_from_graphs(graph_list)

#%% add logDice
g= ig.load('conj\graphs_hrwac\hrwac_conj_centralities.pkl', format='pickle')
dG.graph_add_logDice(g)
ig.write(g, 'conj\graphs_hrwac\hrwac_conj_centralities_logdice.pkl', format='pickle')
#%% load logDice
g= ig.load('conj\graphs_hrwac\hrwac_conj_centralities_logdice.pkl', format='pickle')

#%% similarity
# dG.get_most_similar(dG.second_degree_pruned(g, 'blagost-n', 'logDice', 15, 5, 50, 50, 0, 0, 1, 1), False, 10)
[x for x in  dG.get_most_similar(dG.second_degree_pruned(g, 'napor-n', 'logDice', 15, 15, 0, 0, 0, 0, 1, 1), False, 10)]

#%%%################################# sentiment enrichment

dG.write_odv_sentiment_sentic(g)
ig.write(g, 'conj\graphs_hrwac\hrwac_conj_odv_sentiment_sentic.pkl', format='pickle')
# OVDJE POTENCIJALNO MOŽEMO STAVITI I SentiWords_1.1 rječnik bez veće prilagodbe
# OVDJE POTENCIJALNO MOŽEMO STAVITI I SentiWordNet rječnik uz  prilagodbu na synsetove




#%% Probni graf s gita
import networkx as nx
from matplotlib import pyplot as plt

# construct an example graph G
G = nx.Graph()
cycles = [[1, 2, 4], [1, 3, 4], [1, 5, 6], [3, 4, 40], [5, 6, 52]]
for cycle in cycles:
  nx.add_cycle(G, cycle)
cycles = [[1, 2, 4], [1, 3, 4], [1, 5, 6], [3, 4, 40], [5, 6, 52]]
for cycle in cycles:
    nx.add_cycle(G, cycle)

edges_tuple = [(1, 2, 3),(1, 3, 1.75),(1, 4, 3),(1, 5, 2),(1, 6, 2.5),(1, 7, 0.4),(2, 4, 0.4),(2, 20, 0.8),(2, 21, 2),(2, 22, 0.7),(2, 23, 1),(2, 60, 1.5),(3, 4, 1.4),(3, 40, 0.5),(3, 30, 0.6),(3, 31, 1),(3, 32, 0.4),(4, 40, 1.5),(4, 41, 0.75),(4, 42, 0.4),(5, 6, 1.5),(5, 50, 0.4),(5, 51, 0.8),(5, 52, 1),(6, 52, 0.4),(6, 60, 1.2),(6, 61, 0.5),(7, 70, 2),(7, 71, 0.8),(7, 72, 0.4)]
for edge in edges_tuple:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Set weighted_degree attribute on nodes
weights = nx.get_edge_attributes(G,'weight').values()
weighted_degree =  dict((x,y) for x,y in (G.degree(weight='weight')))
nx.set_node_attributes(G, weighted_degree, "weighted_degree")







#%% LOAD ODV GRAPH ######################################################################
# g= ig.load('conj\graphs_hrwac\hrwac_conj_odv_sentiment_sentic.pkl', format='pickle')
g= ig.load('conj\graphs_hrwac\g_df_npvadjadv_odv.pkl', format='pickle')









#%% subgraf gdje je frequency čvora veća ili jednaka 5
g_n = g.vs.select(freq_ge = 5)
g_n = g.subgraph(g_n)
#%%# koliko čvorova sa odv? i bez odv?
len(g.vs.select(sentic_odv_eq = True)), len(g.vs.select(sentic_odv_eq = False)), len(g.vs.select(sentic_odv_eq = None))
# (17226, 0, 1424176)






#%%################### Array creation
arr = dG.fof_initial_dictionary_list(g,'freq', 15,5,'fofs_vs_array_initial_hrwac_npvadjadv_freq.npy')

#%%
dG.get_fill_steps_and_final_graph(g, arr, 'fill_hrwac_npvadjadv_freq')

#%%
import itertools
import math
#%%%%%%%%%%%%%%%%b logDice problem

logDices=[]
for i in range(0, 1000): 
    e= g.es[i]
    freqR_v1v2 = e['weight']
    v1, v2 = e.source, e.target
    freqN_v1, freqN_v2 = g.vs[v1]['freq'], g.vs[v2]['freq']
    lDice = 14 + math.log2(2*freqR_v1v2/(freqN_v1+freqN_v2))
    logDices.append((v1,v2, freqN_v1, freqN_v2, freqR_v1v2, lDice))
logDices    
# g.es['logDice']= logDices


#%%
e= g.es[663561]
freqR_v1v2 = e['weight']
v1, v2 = e.source, e.target
freqN_v1, freqN_v2 = g.vs[v1]['freq'], g.vs[v2]['freq']
lDice = 14 + math.log2(2*freqR_v1v2/(freqN_v1+freqN_v2))
print(v1,v2, freqN_v1, freqN_v2, freqR_v1v2, lDice)
#%%
14+ np.log2((2*10)/ (6.0+ 1.0))









#%%
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
    print('Graf je tip:', type(g))

    # set edge weight measure to calculate weighted_degree
    edge_measure= 'weight'
    if kwargs.get('weight'):
        edge_measure = kwargs.get('weight')
        
    print('edge_measure', edge_measure)
    g = nx.DiGraph(g) # mora se prebaciti u DiGraph jer ne radi za Multigraph
    g= g.to_undirected() # pretvara ga u neusmjereni graf
    print('Nakon konverzije u DiGraph Graf je tip:', type(g))
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html#networkx.algorithms.cycles.cycle_basis
    cycle = nx.cycle_basis(g) # Will contain: { node_value => sum_of_degrees } 
    print('cycle:', cycle)
    print('broj ciklusa:', len(cycle))
    print('broj čvorova u ciklusima:', [len(x) for x in cycle])

    # weighted_degrees dictionary where keys are gnodes index
    weighted_degrees = g.degree(weight=edge_measure) 
    print('weighted_degrees:', weighted_degrees)
    nx.set_node_attributes(g, weighted_degrees, "weighted_degree")
    print('Postavio weighted_degrees na čvorove')

    # Detekcija ciklusa 
    if cycle:   
        # što skuplja degrees i gdje se to kasnije koristi???? 
        # degrees = {}    
        # dict1 se sastoji od komponente edga i komponente broja ciklusa u kojima sudjeluje npr e: 5
        dict1 = {}
        # mi tu detektiramo weighted degree svakog vrha koji sudjeluje na edgu
        for edge in (nx.edges(g)):
            try:
                print('edge', edge)
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
                    print('edge, in_a_cycle', edge, in_a_cycle)
                    # cicl je par vrhova
                    # prebrojavamo u koliko ciklusa sudjeluje EDGE set(cicl)
                    # counter_edge+=1 
                    for cicl in (cycle):
                        print('cicl ###########', cicl)
                        c=set(cicl) # set edgeva koji sudjeluju u ciklusu {10,53,12,0}
                        set1=set(edge) # edge koji se tretira kao skup {10,53}
                        is_subset = set1.issubset(c)
                        if is_subset==True:
                            counter_edge+=1 
                    print(counter_edge)

                else:
                    print('pass') 
                    pass
                # u dict1 su navedeni bridovi i broj njihovog pojavljivanja u temeljnim cicl
                dict1[edge] = counter_edge
                # u dict1 se p zapisuje na edge a-b i b-a kao ista vrijednost
                # ova linija je bitna za neusmjerene grafove, a ako je usmjeren ona se isključuje
                # treba u slučaju usmjerenih provjeriti i na definiciju ciklusa (cycle)
                dict1[list(itertools.permutations(edge))[1]] = counter_edge
            except:
                pass
        print('dict1', dict1)
    else:
        print('no cycle')

    SLI_importance = []
    # za svaki čvor u grafu 
    for node in nx.nodes(g):
        print('node ###############',node)
        print(f'neigh od {node}', list(g.neighbors(node)))  
        sum = 0
        # nodeWeight_node = g.nodes[node]["weighted_degree"]
        #  weighted_degree od noda
        nodeWeight_node = weighted_degrees[node]

        print('nodeWeight_node', nodeWeight_node)
        # tražimo susjeda za svaki čvor
        for neigh in g.neighbors(node):    
            print(f'neigh od {neigh}') 
            #  uzimamo težinu svakog brida između čvora i susjeda 
            # edge_weight je težina brida između čvora i susjeda    
            edge_weight= g.get_edge_data(node,neigh)['weight']
            print('edge_weight', edge_weight)
            
            # to je weighted degree susjeda od čvora u weighted grafu
            nodeWeight_neigh =  weighted_degrees[neigh]
            print('nodeWeight_neigh', nodeWeight_neigh)
            if not cycle:
                p = 0
            else:
                # p je broj ciklusa u kojima sudjeluje edge
                try:
                    p = dict1[(node, neigh)] # broj ciklusa u kojem sudjeluje edge (node, susjed)
                    print('p', p)
                except:
                    pass
            u = ((nodeWeight_node + nodeWeight_neigh) - (2 * edge_weight))
            print('u', u)
            lambd = p+1 # broj ciklusa +1 u kojima sudjeluje edge
            print('lambd', lambd)
            z = nodeWeight_node / (nodeWeight_node+ nodeWeight_neigh)*edge_weight
            print('z', z)
            # izračun prema definiciji SLI mjere
            I = u*lambd*z
            print('I', I)
            sum = sum + I
            print('sum', sum)
        SLI_importance.append(sum + nodeWeight_node)
        print('sum + nodeWeight_node', sum + nodeWeight_node)
    SLI_importance= pd.Series(SLI_importance)
    print('SLI_importance', SLI_importance)
    print('SLI_importance.sum()', SLI_importance.sum())

    # SLI values non-normalized
    if kwargs.get('normalize') == False:
        SLI_importance_result = SLI_importance
    # SLI values normalized as default
    else:
        SLI_importance_normalized = SLI_importance/SLI_importance.sum()*100
        SLI_importance_result = SLI_importance_normalized 
        print(SLI_importance_result)
    return SLI_importance_result


    

#%%
# sli_importance(fof_g, igraph=True)
import deprelGraph as dG
lemma= 'novac-n'
measure= 'logDice'
# grafic = dG.second_degree_pruned(g.subgraph(g.vs.select(weighted_degree_ge = 5)), lemma, 'logDice', 15, 5, 0, 0, 0, 0, 1, 0.5).to_networkx()
d = dG.second_degree(g,lemma, measure,16,8)
grafic= ig.Graph.TupleList(pd.DataFrame().from_dict(d).itertuples(index=False), directed=False, weights=True, edge_attrs=None)
# grafic = nx.DiGraph(grafic).to_undirected() 
grafic_sli= dG.sli_importance(grafic, weight=measure, igraph=True)
df_grafic= pd.DataFrame([x['name'] for x in grafic.vs()])
df_grafic['SLI']= grafic_sli
print(df_grafic.sort_values(by='SLI', ascending=False)[0:50])

# import matplotlib.pyplot as plt
gn= grafic.to_networkx()
nx.draw(gn)





#%% save the initial arr 
# np.save('conj/fofs_vs_array_initial', arr, allow_pickle=True)
# #load saved array
arr = np.load('fofs_vs_array_initial_hrwac_npvadjadv_freq.npy', allow_pickle=True)




#%%########################### Fill order iteration
fill_steps = dG.get_fill_steps(arr, 'conj/fill_steps_hrwac_npvadjadv_freq')
#%% load
fill_steps = np.load("conj/fill_steps_hrwac_npvadjadv_freq", allow_pickle=True)
# dG.fill_analysis_values(fill_steps)

#%% save the final arr 
# np.save('conj/fofs_vs_array_final', arr, allow_pickle=True)
# #load saved array
# arr = np.load('conj/fofs_vs_array_final.npy', allow_pickle=True)

#%%######## Activate fillings 
g = dG.fill_values(g, arr, fill_steps, 'conj/graphs_final/g_final_a')

#%%# učitaj graf
g= ig.load('conj\graphs_final\g_final_a_graph.pkl', format='pickle')
#%%
# tup=g= np.load('conj\graphs_final\g_final_a_tuples.pkl', allow_pickle=True)
#%%
# g= ig.load(deprel+'/graph_enriched/sentiment_enriched_'+deprel+'_a_graph', format='pickle')
# arr = np.load('conj/fofs_vs_array_initial.npy', allow_pickle=True)
# arr = np.load('conj/fofs_vs_array_final.npy', allow_pickle=True)
# fill_steps = np.load("conj/fill_steps", allow_pickle=True)


#%% final option
g= ig.load('conj/graphs_hrwac/g_df_npvadjadv_odv.pkl', format='pickle')
arr = np.load('fofs_vs_array_initial_hrwac_npvadjadv_freq.npy', allow_pickle=True)

#%%
result = dG.get_fill_steps_and_final_graph(g, arr, 'hrwac_npvadjadv_freq')

#%% naive result
result_naive = dG.get_fill_steps_and_final_graph_naive(g, arr[0:10], 'result_hrwac_npvadjadv_freq_naive')

#%%#########
g= ig.load('result_hrwac_npvadjadv_freq_graph.pkl', format='pickle')
#%%
result_fill_list = np.load('result_hrwac_npvadjadv_freq_fill_list.pkl.npy', allow_pickle=True)
#%%
result_sentiment = np.load('result_hrwac_npvadjadv_freq_sentiment_list.pkl.npy', allow_pickle=True)
#%% g_naive
g = ig.load('results_naive/result_naive_hrwac_npvadjadv_freq_graph_naive.pkl', format='pickle')

#result_hrwac_npvadjadv_freq_naive_graph_naive.pkl
#%% 
node_df = pd.DataFrame({attr: g.vs[attr] for attr in g.vertex_attributes()})
edge_df = pd.DataFrame({attr: g.es[attr] for attr in g.edge_attributes()})