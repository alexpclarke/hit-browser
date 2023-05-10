import librosa
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import warnings
import webbrowser
from pyvis.network import Network
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


# ----- SETTINGS ----- #

valid_types = [".wav", ".flac"]
out_filename = 'hit-browser.html'
n_matches = 2
tremolo_wsize = 8
fit_degree = 1
plot = False
stabilization_iterations = 50




# ----- HELPER FUNCTIONS ----- #

# Linearly normalize any data to between 0 and 1.
def normalize_linear(arr):
  return ((arr - max(np.min(arr), 0)) / np.max(arr))

# Print an error message and/or how to use this program.
def print_usage(msg):
  if (msg): print("ERROR: " + msg)
  print("USAGE: python3 hb.py <path to hit library>")
  quit()




# ----- READ ARGUMENTS ----- #

# Check the number of aruments provided.
if (len(sys.argv) != 2):
  print_usage("Invalid number of arguments.")

# Check that the path is valid.
if os.path.isdir(sys.argv[1]):
  hit_dir = os.path.abspath(sys.argv[1])
else:
  print_usage("Not a valid path.")




# ----- LOAD FILE DATA -----#

print('loading files...')

# Inititialize the array that will store the data about each hit file.
hits = np.array([])

# Find the path to all the valid audio files.
for root, dirs, files in os.walk(hit_dir, topdown = True, followlinks = True):
  for file in files:
    # Get the file's absolute path.
    path = root + "/" + file

    # Get the file's extension and ignore the files without a valid extension.
    ext = re.findall("\.[a-z,0-9]+$", file)
    if len(ext) == 0 or ext[0] not in valid_types:
      continue

    # Turn off warnings.
    warnings.filterwarnings("ignore")

    # Load the file as a floating point time series (time x amplitude) and trim
    # the leading and trailing silence.
    try:
      data_orig, sr = librosa.load(path, sr = None)
      data = librosa.effects.trim(data_orig, top_db = 5)[0]
    except:
      print("WARNING: Failed to load: " + path)
      continue

    # Turn warnings back on.
    warnings.resetwarnings()

    # Ignore files with no data
    if (len(data) == 0): 
      print("WARNING: Failed to load (no data): " + path)
      continue

    # Add the file to the list of files.
    hits = np.append(hits, {
      "path": path[len(hit_dir):],
      "data": data,
      "sr": sr,
      "props": []
    })

# Make sure that there were some hits loaded.
if (len(hits) == 0):
  print_usage("No hits found.")




# ----- DURATION ----- #

print('calculating durations...')

# Set the file's duration in samples.
for hit in hits:
  hit['duration'] = len(hit['data'])

# Normalize and weight the values.
hit_duration = np.array([hit['duration'] / hit['sr'] for hit in hits])
hit_duration = normalize_linear(hit_duration)
for (i, hit) in enumerate(hits):
  hit['props'].append(hit_duration[i])




# ----- PITCH ----- #

print('analyzing pitches...')

framesize = 256

# Calculate the STFT of the file and power over time.
for hit in hits:
  hit['stft'] = np.abs(librosa.stft(
    y = hit['data'],
    n_fft = framesize,
    hop_length = 1
  ))

  hit['power'] = librosa.amplitude_to_db(
    np.sum(hit['stft'], axis = 0),
    ref = np.max
  )
  hit['power'] = hit['power'] - np.min(hit['power'])

# Determine the average pitch of the file.
# (The pitch bin with the highest average power)
for hit in hits:
  total_magnitude = np.sum(hit['stft'], axis = 0)
  total_magnitude[total_magnitude == 0] = 1
  bin_weights = hit['stft'] / total_magnitude
  bin_vals = (np.arange(int(framesize / 2) + 1)[:, np.newaxis] + 1)
  pitches = np.sum(bin_weights * bin_vals, axis = 0) - 1
  pitches = pitches

  hit['pitch_line'] = np.polyfit(
    np.arange(len(pitches)) / hit['sr'],
    pitches,
    fit_degree,
    full = True
  )

# Normalize the values.
residuals = np.array([hit['pitch_line'][1][0] for hit in hits])
residuals = np.max(residuals) - residuals
residuals = normalize_linear(residuals)

for i in range(fit_degree + 1):
  c_i = np.array([hit['pitch_line'][0][i] for hit in hits])
  c_i = normalize_linear(c_i) * (2 ** (fit_degree - i))
  for (j, hit) in enumerate(hits):
    if i == 0:
      hit['props'].append(c_i[j] * residuals[j])
    else:
      hit['props'].append(c_i[j])




# ----- ATTACK ----- #

print('calculating attack...')

# Calculate the attack. (Time between 0% and 100% power)
for hit in hits:
  hit['attack'] = np.argmax(hit['power'])

# Normalize and weight the values.
hit_attack = np.array([hit['attack'] / hit['sr'] for hit in hits])
hit_attack = normalize_linear(hit_attack)
for (i, hit) in enumerate(hits):
  hit['props'].append(hit_attack[i])




# ----- TREMOLO ----- #

print('calculating tremolo...')

for hit in hits:
  # Find all the peaks.
  hit['peaks'] = [0]
  i = 0
  while (i < len(hit['data'])):
    win_min = max(i - tremolo_wsize, 0)
    win_max = min(i + tremolo_wsize + 1, len(hit['data']))
    peak_window = hit['power'][win_min : win_max]
    if (np.argmax(peak_window) == i - win_min):
      range_min = hit['power'][hit['peaks'][-1] : i + 1].min()
      range_max = hit['power'][hit['peaks'][-1] : i + 1].max()
      if (range_max - range_min > 2):
        hit['peaks'].append(i)
      elif (hit['power'][hit['peaks'][-1]] < hit['power'][i]):
        hit['peaks'].append(i)
    i += 1
  hit['peaks'].append(i - 1)

  # The gap between peaks.
  hit['peak_diffs'] = np.ediff1d(np.array(hit['peaks']) / hit['sr'])

  # If there are not enough gaps to approximate
  if len(hit['peak_diffs']) < fit_degree + 1:
    hit['tremolo_line'] = None
    continue

  # Relationship between the gap in peaks.
  hit['tremolo_line'] = np.polyfit(
    np.arange(len(hit['peak_diffs'])), 
    hit['peak_diffs'], 
    fit_degree, 
    full = True
  )

# Normalize the values.
residuals = np.array([hit['pitch_line'][1][0] for hit in hits])
residuals = np.max(residuals) - residuals
residuals = normalize_linear(residuals)

for i in range(fit_degree + 1):
  c_i = np.array([(hit['tremolo_line'][0][i] if hit['tremolo_line'] else 0) for hit in hits])
  c_i = normalize_linear(c_i) * (2 ** (fit_degree - i))
  # c_i = normalize_linear(c_i)
  for (j, hit) in enumerate(hits):
    if i == 0:
      hit['props'].append(c_i[j] * residuals[j])
    else:
      hit['props'].append(c_i[j])



# ----- PLOT ANALYSIS ----- #

# Plot hit.
def plot_hit(hit, filename):
  fig, axs = plt.subplots(3)
  time_ticks = np.arange(0, hit['duration'] + 1, hit['sr'] / 100)

  # ----- Plot 1 ----- #

  axs[0].plot(np.arange(hit['duration'] + 1), hit['power'])
  axs[0].set_xticks(time_ticks, time_ticks / hit['sr'])
  axs[0].margins(x = 0)

  for peak in hit['peaks']:
    axs[0].axvline(x = peak, color = 'red')
  
  axs[0].plot([0, hit['attack'] / hit['sr']], [hit['power'][0], np.max(hit['power'])], color = 'green')

  # ----- Plot 2 ----- #

  axs[1].plot(np.arange(len(hit['peak_diffs'])), hit['peak_diffs'], 'o')
  if (hit['tremolo_line']):
    x = np.linspace(0, len(hit['peak_diffs']))
    y = 0
    for (i, c) in enumerate(hit['tremolo_line'][0]):
      y += c * (x ** (fit_degree - i))
    axs[1].plot(x, y, '-r', label='y=2x+1')

  # ----- Plot 3 ----- #

  axs[2].pcolormesh(hit['stft'])
  axs[2].set_xticks(time_ticks, time_ticks / hit['sr'])
  axs[2].set_yticks([0, framesize / 2 + 1], [0, int(hit['sr'] / 2000)])
  axs[2].axis(ymin = 0, ymax = framesize / 2 + 1)
  if (hit['pitch_line']):
    x = np.linspace(0, (hit['duration'] + 1))
    y = 0
    for (i, c) in enumerate(hit['pitch_line'][0]):
      y += c * (x ** (fit_degree - i))
    axs[2].plot(x, y, '-r', label='y=2x+1')

  fig.suptitle(hit['path'])
  plt.savefig('plots/' + filename + '.png')
  plt.close()

if (plot):
  print('generating plots...')
  for (i, hit) in enumerate(hits):
    plot_hit(hit, str(i))




# ----- CALCULATE EDGES ----- #

print('calculating edges...')

# Calculate the position of the nodes.
hit_props = np.array([hit['props'] for hit in hits])
pca = PCA(n_components = 2)
hit_pos = pca.fit_transform(hit_props)

# Calculate the position edges.
hit_matches = kneighbors_graph(
  hit_pos,
  n_neighbors = n_matches,
  mode = 'distance'
).toarray()




# ----- CREATE NETWORKX NETWORK ----- #

print('creating graph...')

# Set up the graph.
hit_graph = nx.Graph()

# Add the nodes to the graph.
for (i, hit_file) in enumerate(hits):
  hit_graph.add_node(i, path = hit_file['path'], size = 10)


# Add the edges to the graph.
for (i_1, matches) in enumerate(hit_matches):
  for (i_2, dist) in enumerate(matches):
    if (dist):
      hit_graph.add_edge(i_1, i_2, weight = (np.max(hit_matches) - dist) / 4)

# Print out the number of nodes and edges.
print(hit_graph)




# ----- STABILIZE NETWORK ----- #

print('stabilizing graph...')

stable_positions = nx.spring_layout(
  hit_graph,
  dim = 2,
  k = 0.3,
  pos = {i: pos for (i, pos) in enumerate(zip(hit_pos[:,0], hit_pos[:,1]))},
  iterations = stabilization_iterations,
  weight = 'weight',
  scale = 1000
)

for (node, nodedata) in hit_graph.nodes.items():
  hit_graph.nodes[node]['x'] = stable_positions[node][0]
  hit_graph.nodes[node]['y'] = stable_positions[node][1]




# ----- CREATE PYVIZ NETWORK ----- #

print('rendering graph...')

# Initialize the graph.
net = Network()

# Set the options for the graph.
net.set_template('./template.html')
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
    "enabled": false
  }
}
""")
net.heading = """
{
  "root_path": "%s"
}
""" % (hit_dir)

# Import the nodes from the NetworkX graph.
net.from_nx(hit_graph)

# Write the graph to an HTML file and open it.
net.show(out_filename)
webbrowser.open('file:///' + os.getcwd() + '/' + out_filename)
