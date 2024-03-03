#%%
import dash
from dash import html, dcc, Output, Input
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_cytoscape as cyto
import networkx as nx
import igraph as ig
import community as community_louvain
import gc
import pickle

#%%%%%%%%%%%%%%%%

# Simplified function to convert an igraph graph to a NetworkX graph
def convert_igraph_to_networkx(igraph_graph):
    nx_graph = nx.DiGraph() if igraph_graph.is_directed() else nx.Graph()
    
    # Add nodes with attributes
    for vertex in igraph_graph.vs:
        nx_graph.add_node(vertex.index, **vertex.attributes())
    
    # Add edges, optionally with attributes if there are any
    for edge in igraph_graph.es:
        attrs = edge.attributes()
        # If the edge has attributes, add them, else just add the edge
        if attrs:
            nx_graph.add_edge(edge.source, edge.target, **attrs)
        else:
            nx_graph.add_edge(edge.source, edge.target)
    
    return nx_graph

# Loading and converting the graph
print('Loading and converting graph...')
g = ig.load('result_hrwac_freq_graph_nouns.pkl', format='pickle')
G = convert_igraph_to_networkx(g)
del g
gc.collect()
print('Graph loaded and converted.')




# version with uploading directly to NetworkX
# G = nx.read_gpickle('result_hrwac_freq_graph_nouns_NetworkX.gpickle')


# def load_graph():
#     return nx.read_gpickle('result_hrwac_freq_graph_nouns_NetworkX.gpickle')
# #lazy loading
# G = None
# def get_graph():
#     global G
#     if G is None:
#         print('Loading and converting graph...')
#         G = load_graph()
#         print('Graph loaded and converted.')
#     return G

# G = get_graph()

#         
#%%%%%%%%
def find_node_by_unique_name(graph, name):
    for node, attr in graph.nodes(data=True):
        if attr.get('name') == name:
            return node
    return None 

def get_top_x_neighbors(graph, node, number_of_neighbors):
    # Get all edges connected to 'node' with their weights
    edges = [(n, graph[node][n]['logDice']) for n in graph.neighbors(node)]
    # Sort the list of tuples based on weight in descending order
    sorted_edges = sorted(edges, key=lambda x: x[1], reverse=True)
    # Extract the top number_of_neighbors (or fewer, if node has fewer than x neighbors)
    top_neighbors = [edge[0] for edge in sorted_edges[:number_of_neighbors]]
    return top_neighbors

# def build_subgraph_from_top_neighbors(graph, name, number_of_neighbors):
#     node = find_node_by_unique_name(graph, name)
#     top_neighbors = get_top_x_neighbors(graph, node, number_of_neighbors)
#     # Include the original node in the list for the subgraph
#     top_neighbors.append(node)
#     # Create the subgraph
#     subgraph = graph.subgraph(top_neighbors)
#     return subgraph


def build_subgraph_from_top_neighbors(graph, name, number_of_neighbors_1st, number_of_neighbors_2nd):
    node = find_node_by_unique_name(graph, name)
    if node is None:
        return nx.Graph()  # Return an empty graph if the node is not found

    # First order
    top_neighbors = get_top_x_neighbors(graph, node, number_of_neighbors_1st)
    
    # Second order
    for neighbor in top_neighbors.copy():  # Use copy() to avoid modifying the list during iteration
        neighbors_of_neighbor = get_top_x_neighbors(graph, neighbor, number_of_neighbors_2nd)
        top_neighbors.extend(neighbors_of_neighbor)
    
    # Remove duplicates and ensure the original node is included
    unique_neighbors = list(set(top_neighbors + [node]))
    
    # Create the subgraph
    subgraph = graph.subgraph(unique_neighbors)
    print(unique_neighbors)
    return subgraph


#%%
# subgraph = build_subgraph_from_top_neighbors(G, 'blagostanje-n', 10,2)



def generate_plotly_graph_from_subgraph(subgraph):
    # Assuming 'subgraph' is your NetworkX graph of interest
    pos = nx.spring_layout(subgraph, dim=3)  # Use 3D layout

    # Edge traces
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in subgraph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # Add None to create a segment
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Node traces
    node_x = []
    node_y = []
    node_z = []
    node_text = []  # Labels for nodes
    hover_text = []  # Hover texts
    for node in subgraph.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_name = subgraph.nodes[node].get('name', str(node))
        node_text.append(node_name)
        
        # Assuming you have attributes to display on hover
        hover_text.append(f'{node_name}: Additional Info')

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=node_text,
        hoverinfo='text',
        hovertext=hover_text,
        textposition="top center",
        marker=dict(
            size=10,
            color='#007bff',  # Or any logic to color nodes
            line_width=2))

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        scene=dict(
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                    ))

    return fig  # Return the figure object to be used in the callback



#%%


# Your existing code to convert igraph to networkx, load the graph, and other functions
# ...

app = dash.Dash(__name__)

# Update the layout to include an input field and a div for the plotly graph
app.layout = html.Div([
    html.Div([  # This div contains the input controls
        dcc.Input(id='lexeme-input', type='text', placeholder='Enter lempos name', style={'marginRight': '10px'}),
        html.Button('Submit', id='submit-val', n_clicks=0)
    ], style={'marginBottom': '20px'}),  # Add some space below the input controls
    html.Div([  # This div will contain the graph
        dcc.Graph(id='network-graph')
    ])
], style={'padding': '20px'})  # Add some padding around the entire layout for aesthetics

@app.callback(
    Output('network-graph', 'figure'),
    [Input('submit-val', 'n_clicks')],
    [State('lexeme-input', 'value')]
)
def update_graph(n_clicks, value):
    if value:
        node = find_node_by_unique_name(G, value)
        if node is not None:
            subgraph = build_subgraph_from_top_neighbors(G, value, 15, 5)
            fig = generate_plotly_graph_from_subgraph(subgraph)  # Call your function
            return fig  # Return the figure object to be rendered inside the dcc.Graph
        else:
            return go.Figure()  # Return an empty figure if the node is not found or there's no input
    else:
        return go.Figure()


if __name__ == '__main__':
    app.run_server(debug=True)


#%%


# import dash
# from dash import dcc, html, Input, Output
# import dash_cytoscape as cyto
# import networkx as nx
# import plotly.graph_objs as go

# # Assuming G is your main NetworkX graph initialized somewhere in your code

# # Initialization of your Dash app
# app = dash.Dash(__name__)
# app.layout = html.Div([
#     dcc.Input(id='lexeme-input', type='text', placeholder='Enter lexeme name'),
#     html.Button('Submit', id='submit-val', n_clicks=0),
#     cyto.Cytoscape(id='cytoscape-ego', style={'width': '100%', 'height': '400px'})
# ])

# @app.callback(
#     Output('cytoscape-ego', 'elements'),
#     [Input('submit-val', 'n_clicks')],
#     [dash.dependencies.State('lexeme-input', 'value')]
# )
# def update_output(n_clicks, value):
#     if value is not None and G.has_node(value):
#         # Assuming build_subgraph_from_top_neighbors is defined elsewhere and returns a NetworkX graph
#         subgraph = build_subgraph_from_top_neighbors(G, value, 10, 2)
        
#         elements = []

#         # Edges
#         for edge in subgraph.edges():
#             source, target = edge
#             elements.append({
#                 'data': {'id': f'{source}{target}', 'source': source, 'target': target}
#             })

#         # Nodes
#         for node in subgraph.nodes():
#             node_data = subgraph.nodes[node]
#             # Adapt additional properties as needed
#             elements.append({
#                 'data': {
#                     'id': node,
#                     'label': node_data.get('name', str(node)),
#                     # Include any additional properties you want from the node attributes
#                     'polarity_value': node_data.get('polarity_value', 'N/A'),
#                     'sentic_odv': node_data.get('sentic_odv', 'N/A'),
#                     'adv_cert': node_data.get('adv_cert', 'N/A')
#                 }
#             })
        
#         return elements
#     else:
#         return []

# if __name__ == '__main__':
#     app.run_server(debug=True)

# %%
