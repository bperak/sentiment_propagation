#%%
# import deprelGraph as dG
import sys
import os
import time
import json
import igraph as ig
import numpy as np
import pandas as pd
import pickle
#%%
"""Download result_hrwac_npvadjadv_freq_graph.pkl graph file from:
https://uniri-my.sharepoint.com/:u:/g/personal/bperak_uniri_hr/Eeb_IrL1NXZCtp8Nj0GyJSQBZRAi__poEWF4eczzrVihog?e=OLbX2l
"""

# 1. Load graph
print('Loading graph...')

# with sentic values
# g= ig.load('hrwac_conj_odv_sentiment_sentic.pkl', format='pickle')

# with calculated values
g= ig.load('result_hrwac_npvadjadv_freq_graph.pkl', format='pickle')
print('Graph loaded.')

# attributes of the graph
print(g.vs.attributes())
# ['name', 'freq', 'degree', 'weighted_degree', 'polarity_value', 'pleasantness', 'attention', 'sensitivity', 'aptitude', 'sentic_odv', 'aptitudes', 'adv_cert']
#%%
# how many nodes are there?
print('Number of nodes:', len(g.vs))
# Number of nodes: 990327
print('Number of nouns:', len([node for node in g.vs if node["name"].endswith("-n")] ))
print('Number of adjectives :', len([node for node in g.vs  if node["name"].endswith("-a")] ))
print('Number of adverbs :', len([node for node in g.vs  if node["name"].endswith("-r")] ))
print('Number of verbs :', len([node for node in g.vs if node["name"].endswith("-v")] ))


# how many edges are there?
print('Number of edges:', len(g.es))

# how many nodes are there with sentic values?
print('Number of nodes with sentic values:', len([node for node in g.vs if node['sentic_odv']==True]))
print('Number of nouns with sentic values:', len([node for node in g.vs if node['sentic_odv']==True if node["name"].endswith("-n")] ))
print('Number of adjectives with sentic values:', len([node for node in g.vs if node['sentic_odv']==True if node["name"].endswith("-a")] ))
print('Number of adverbs with sentic values:', len([node for node in g.vs if node['sentic_odv']==True if node["name"].endswith("-r")] ))
print('Number of verbs with sentic values:', len([node for node in g.vs if node['sentic_odv']==True if node["name"].endswith("-v")] ))

# how many nodes are there with calculated values?
print('Number of nodes with calculated values:', len([node for node in g.vs if node['sentic_odv']==False]))
print('Number of nouns with calculated values:', len([node for node in g.vs if node['sentic_odv']==False if node["name"].endswith("-n")]))
print('Number of adjectives with calculated values:', len([node for node in g.vs if node['sentic_odv']==False if node["name"].endswith("-a")]))
print('Number of adverbs with calculated values:', len([node for node in g.vs if node['sentic_odv']==False if node["name"].endswith("-r")] ))
print('Number of verbs with calculated values:', len([node for node in g.vs if node['sentic_odv']==False if node["name"].endswith("-v")]))

# how many nodes are there without calculated values?
print('Number of nodes without calculated values:', len([node for node in g.vs if node['sentic_odv']==None ]))
print('Number of nouns without calculated values:', len([node for node in g.vs if node['sentic_odv']==None if node["name"].endswith("-n")]))
print('Number of adjectives without calculated values:', len([node for node in g.vs if node['sentic_odv']==None if node["name"].endswith("-a")]))
print('Number of adverbs without calculated values:', len([node for node in g.vs if node['sentic_odv']==None if node["name"].endswith("-r")]))
print('Number of verbs without calculated values:', len([node for node in g.vs if node['sentic_odv']==None if node["name"].endswith("-v")]))


#%% save nodes list with their attributes to csv with tab separator
def save_nodes_to_csv(g, pos):
    # Extract node attributes
    node_attrs = g.vs.attributes()

    # Filter nodes whose names end with "-n"
    filtered_nodes = [v for v in g.vs if v['name'].endswith('-n')]

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Add node IDs to the DataFrame
    df['id'] = [v.index for v in filtered_nodes]

    # Add other node attributes to the DataFrame
    for attr in node_attrs:
        df[attr] = [v[attr] for v in filtered_nodes]
    # You can save it to a CSV file if needed
    filename = 'result_hrWac_lexical_graph_nodes'+f'{pos}'+'.csv'
    df.to_csv(filename, index=False)
    print(f'Nodes saved to {filename}')
save_nodes_to_csv(g, '-v')

#%%
def profiling_edges_by_pos(g, pos):
    # Find all vertices where the name ends with '-n'
    filtered_vertex_indices = [v.index for v in g.vs if v['name'].endswith('-'+pos)]
    # Create a subgraph with only the filtered vertices
    subgraph = g.subgraph(filtered_vertex_indices)
    # Now count the edges in the subgraph
    edge_count = len(subgraph.es)
    print(f'Number of edges where both nodes end with -{pos}: {edge_count}')
    
profiling_edges_by_pos(g, 'n')
profiling_edges_by_pos(g, 'a')
profiling_edges_by_pos(g, 'r')
profiling_edges_by_pos(g, 'v')
    
#%%

#%%
def profiling_by_pos(g):
    nouns = [node for node in g.vs if node["name"].endswith("-n")]
    adjectives = [node for node in g.vs if node["name"].endswith("-a")]
    adverbs = [node for node in g.vs if node["name"].endswith("-r")]
    verbs = [node for node in g.vs if node["name"].endswith("-v")]
    print('Nouns',len(nouns),'adjectives', len(adjectives),'adverbs', len(adverbs),'verbs', len(verbs))
profiling_by_pos(g)    
#%%
# 2. Profiling of calculated values
print('Number of calculated nodes',len([node['adv_cert'] for node in g.vs if node['sentic_odv']==False]))
adv_cert = [node['adv_cert'] for node in g.vs if node['sentic_odv']==False]
adv_cert_s = pd.Series(adv_cert)
# 
adv_cert_values= adv_cert_s.value_counts().sort_index(ascending=False).rename_axis('adv_cert').reset_index(name='count')
adv_cert_values
#%%
adv_cert_values.to_csv('deprel_Evaluation_adv_cert_distribution.tsv', header=True, sep='\t') 

# %%
adv_cert_values.plot(kind='bar', figsize=(15,5), logy=True)

# %%
# Step 1: Filter nodes where the name ends with '-n'
g_nouns_nodes = [v.index for v in g.vs if v["name"].endswith("-n")]

# Step 2: Create a subgraph with these nodes
subgraph_nouns = g.subgraph(g_nouns_nodes)

subgraph_nouns.write_pickle('final_g_nouns.pkl')
# %%
