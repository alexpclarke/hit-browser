# import tkinter as tk
# import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import librosa
import os # for reading directories
import re # for regex

# ----- Settings ----- #
hit_lib = "/Library/Audio/Sounds/Drum Kits"
valid_file_types = [".wav", ".flac"]
resample_rate = 22050

# ----- Load Files -----#
def load_files(path):
  arr = []

  path_walk = os.walk(path, topdown=True, followlinks=True)
  for root, dirs, files in path_walk:
    for file in files:
      # Get the absolute path.
      file_path = root + "/" + file

      # Ignore the files that dont have a valid extension.
      file_ext = re.findall("\.[a-z,0-9]+$", file_path)
      if len(file_ext) == 0 or file_ext[0] not in valid_file_types:
        continue

      # Load the file data.
      try:
        file_data, file_sr = librosa.load(file_path, sr=None)
      except:
        print("Failed to load: " + file_path)
        continue

      # Add the file to the list of files.
      arr.append({
        "name": file,
        "path": file_path,
        "ext": file_ext[0],
        "sr": file_sr,
        "data": file_data
      })

  return arr



# ----- Main ----- #
hit_files = load_files(hit_lib)
# print(hit_files)

# Set up the graph.
G = nx.Graph()

# Add the nodes to the graph.
for f in hit_files:
  G.add_node(f["name"], path=f['path'], ext=f['ext'])

# Add the edges to the graph.

# Calculate the positions for the nodes.
pos = nx.spring_layout(G, k = 0.5, iterations = 100)
for n, p in pos.items():
  G.nodes[n]['pos'] = p

# Calculate the positions of the edges.
edge_x = []
edge_y = []
for edge in G.edges():
  x0, y0 = G.nodes[edge[0]]['pos']
  x1, y1 = G.nodes[edge[1]]['pos']
  edge_x.append(x0)
  edge_x.append(x1)
  edge_x.append(None)
  edge_y.append(y0)
  edge_y.append(y1)
  edge_y.append(None)

edge_trace = go.Scatter(
  x=edge_x, y=edge_y,
  line=dict(width=0.5, color='#888'),
  hoverinfo='none',
  mode='lines')

node_x = []
node_y = []
for node in G.nodes():
  x, y = G.nodes[node]['pos']
  node_x.append(x)
  node_y.append(y)

node_trace = go.Scatter(
  x = node_x, y = node_y,
  mode = 'markers',
  hoverinfo = 'text',
  marker = dict(
    showscale = False,
    # colorscale options
    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
    colorscale = 'YlGnBu',
    reversescale = True,
    color = [],
    size = 10,
    # colorbar=dict(
    #     thickness=15,
    #     title='Node Connections',
    #     xanchor='left',
    #     titleside='right'
    # ),
    line_width=2
  )
)

# Set the color and hover text for each node.
node_color = []
node_text = []
for (node_name, node_data) in G.nodes.items():
  node_color.append(valid_file_types.index(node_data["ext"]))
  node_text.append(node_name)

node_trace.marker.color = node_color
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
  layout = go.Layout(
    title = 'Drum Hit Similarity',
    titlefont_size=16,
    showlegend = False,
    hovermode = 'closest',
    margin = dict(b=20,l=5,r=5,t=40),
    xaxis = dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis = dict(showgrid=False, zeroline=False, showticklabels=False)
  )
)
fig.show()