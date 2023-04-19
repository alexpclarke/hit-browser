import networkx as nx
import librosa
import os
import re
import webbrowser
from pyvis.network import Network
import warnings

# ----- Settings ----- #
hit_lib = "/Library/Audio/Sounds/Drum Kits/808_drum_kit"
valid_file_types = [".wav", ".flac"]
out_filename = 'net.html'

# ----- Load Files -----#
def load_files(path):
  arr = []

  path_walk = os.walk(path, topdown = True, followlinks = True)
  for root, dirs, files in path_walk:
    for file in files:
      # Get the absolute path.
      file_path = root + "/" + file

      # Ignore the files that dont have a valid extension.
      file_ext = re.findall("\.[a-z,0-9]+$", file_path)
      if len(file_ext) == 0 or file_ext[0] not in valid_file_types:
        continue

      # Load the file data.
      warnings.filterwarnings("ignore")
      try:
        file_data, file_sr = librosa.load(file_path, sr = None)
        file_data_trim, file_data_trim_index = librosa.effects.trim(file_data)
      except:
        print("Failed to load: " + file_path)
        continue
      warnings.resetwarnings()

      # Add the file to the list of files.
      arr.append({
        "name": file,
        "path": file_path[len(path):],  #Make this relative to the root
        "ext": file_ext[0],
        "sr": file_sr,
        "dur": librosa.get_duration(y = file_data_trim, sr = file_sr),
        "data": file_data_trim
      })

  return arr

# ----- Greating NetworkX Graph  ----- #
# Load and analyze all of the files.
hit_files = load_files(hit_lib)

# Set up the graph.
hit_graph = nx.Graph()

# Add the nodes to the graph.
for (i, hit_file) in enumerate(hit_files):
  hit_graph.add_node(
    i,
    label = None,
    name = hit_file['name'],
    path = hit_file['path'],
    ext = hit_file['ext'],
    size = 40
  )

# Add the edges to the graph.
for i1 in range(0, len(hit_files)):
  for i2 in range(i1 + 1, len(hit_files)):
    # Check the percent difference in length.
    dur1 = hit_files[i1]['dur']
    dur2 = hit_files[i2]['dur']
    perc_diff = abs(dur1 - dur2) / ((dur1 + dur2) / 2)
    if (perc_diff < 0.005):
      hit_graph.add_edge(i1, i2)

net = Network(cdn_resources="remote")

net.set_template(os.getcwd() + '/template.html')
net.set_options("""
var options = {
  "autoResize": true,
  "locale": "en",
  "edges": {
    "chosen": false
  },
  "layout": {
    "improvedLayout": false
  },
  "interaction": {
    "dragNodes": false
  },
  "physics": {
    "enabled": false,
    "solver": "barnesHut",
    "barnesHut": {
      "theta": 1,
      "gravitationalConstant": -2000,
      "centralGravity": 0.3,
      "springLength": 95,
      "springConstant": 0.04,
      "damping": 0.09,
      "avoidOverlap": 1
    }
  }
}
""")

# nodes, edges, heading, height, width, options = net.get_network_data()
# print(heading)

net.from_nx(hit_graph)

# net.generate_html(out_filename)
net.show(out_filename)
webbrowser.open('file:///' + os.getcwd() + '/' + out_filename)
