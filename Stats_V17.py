#%% 
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import networkx as nx
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from itertools import combinations
from networkx.algorithms.community import greedy_modularity_communities
import community as community_louvain

#%% --- DATA ---

file_path = "DATA_COMPRENDRE.xlsx"
data = pd.read_excel(file_path)

#%% --- CORRELATIONS --- 
tasks = ['PHO', 'LEX-R', 'VOC', 'SYN', 'MEM_V', 'UP_V', 'INHIB_V', 'MEM_NV', 'UP_NV', 'INHIB_NV']
labels = {'PHO': 'PHO', 'LEX-R': 'LEX-R', 'VOC': 'VOC', 'SYN': 'SYN', 'MEM_V': 'MEM V', 'UP_V': 'UP V', 
          'INHIB_V': 'INHIB V', 'MEM_NV': 'MEM NV', 'UP_NV': 'UP NV', 'INHIB_NV': 'INHIB NV'}

colors = {'PHO': '#E68E00', 'LEX-R': '#E68E00', 'VOC': '#E68E00', 'SYN': '#E68E00', 
          'MEM_V': '#0080D5', 'UP_V': '#0080D5', 'INHIB_V': '#0080D5',
          'MEM_NV': '#9000CB', 'UP_NV': '#9000CB', 'INHIB_NV': '#9000CB'}

class_titles = {'GS': 'Kindergarten', 'CP': '1st grade', 'CE1': '2nd grade', 'CE2': '3rd grade'}

# Function for calculating partial correlation matrices 
def calculate_partial_correlations(data, tasks, control_variable):
    correlations = {}
    for i, task1 in enumerate(tasks):
        for task2 in tasks[i+1:]:
            result = pg.partial_corr(data=data, x=task1, y=task2, covar=[control_variable], method='spearman')
            correlations[(task1, task2)] = (result['r'].values[0], result['p-val'].values[0])
    return correlations

# Plot heatmaps
def plot_heatmap(corr_matrix, p_matrix, cls, tasks, labels, colors, class_titles, save_path=None):
    plt.figure(figsize=(10, 8))
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    masked_corr = np.ma.masked_where(mask, corr_matrix)
    heatmap = plt.imshow(masked_corr, cmap='YlGnBu', interpolation='nearest')
    cbar = plt.colorbar(heatmap, pad=0.05)
    cbar.ax.set_title("Spearman's\nCorrelation", fontsize=12, ha='center')
    plt.clim(0, 1)

    plt.title(class_titles.get(cls, cls), fontsize=24, fontweight='bold', y=1.05)
    ax = plt.gca()
    
    ax.set_xticks(range(len(tasks)))
    ax.set_yticks(range(len(tasks)))
    ax.set_xticklabels([labels[task] for task in tasks], fontsize=14, fontweight='bold', rotation=45)
    ax.set_yticklabels([labels[task] for task in tasks], fontsize=14, fontweight='bold')
    
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(colors.get(tasks[i]))
        ax.get_yticklabels()[i].set_color(colors.get(tasks[i]))

    save_path = f"heatmap_corr_p_{cls}.png"

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                significance = "*" if p_val < 0.001 else "*" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                if corr_val < 0 or p_val >= 0.05:  
                    plt.text(j, i, f"{corr_val:.2f}\n{significance}", 
                             ha='center', va='center', color='#575757', fontsize=14, fontweight='normal')
                else: 
                    plt.text(j, i, f"{corr_val:.2f}\n{significance}", 
                             ha='center', va='center', color='black', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def process_all_GRADE(data, tasks, control_variable):
    correlations = {}
    p_matrices = {} 

    for cls in data['GRADE'].unique():
        print(f"\nProcessing Class {cls}:")
        class_data = data[data['GRADE'] == cls]  
        class_correlations = calculate_partial_correlations(class_data, tasks, control_variable)


        corr_matrix = pd.DataFrame(index=tasks, columns=tasks, dtype=float)
        p_matrix = pd.DataFrame(index=tasks, columns=tasks, dtype=float)

        task_pairs, p_values = list(class_correlations.keys()), [p for _, p in class_correlations.values()]
        corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

        for idx, (task1, task2) in enumerate(task_pairs):
            corr_matrix.loc[task1, task2] = corr_matrix.loc[task2, task1] = class_correlations[(task1, task2)][0]
            p_matrix.loc[task1, task2] = p_matrix.loc[task2, task1] = corrected_p_values[idx]

        correlations[cls] = corr_matrix
        p_matrices[cls] = p_matrix  


        save_path = f"heatmap_corr_partielle_{cls}.png"
        plot_heatmap(corr_matrix, p_matrix, cls, tasks, labels, colors, class_titles, save_path)

    return correlations, p_matrices  

correlations, p_matrices = process_all_GRADE(data, tasks, 'MEAN_PARENTS_STUDIES')


#%% """Textual display of correlations with significance for each GRADE''

def print_textual_correlations(data, tasks, control_variable):
    for cls in data['GRADE'].unique():
        print(f"\nResults for Class {class_titles.get(cls, cls)}:")
        class_data = data[data['GRADE'] == cls]  
        class_correlations = calculate_partial_correlations(class_data, tasks, control_variable)
        task_pairs, p_values = list(class_correlations.keys()), [p for _, p in class_correlations.values()]
        corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

        for idx, (task_pair, (corr_val, _)) in enumerate(class_correlations.items()):
            task1, task2 = task_pair
            corrected_p_val = corrected_p_values[idx]
            significance = "***" if corrected_p_val < 0.001 else "**" if corrected_p_val < 0.01 else "*" if corrected_p_val < 0.05 else ""
            
            print(f"Correlation between {task1} and {task2}: r = {corr_val:.2f}, p = {corrected_p_val:.4f} {significance}")

print_textual_correlations(data, tasks, 'MEAN_PARENTS_STUDIES')

# %% --- GRAPH THEORY ---
SIGNIFICANCE_THRESHOLD = 0.05

LABELS = {
    'PHO': 'PHO', 'LEX-R': 'LEX-R', 'VOC': 'VOC', 'SYN': 'SYN',
    'MEM_V': 'MEM\nV', 'UP_V': 'UP\nV', 'INHIB_V': 'INHIB\nV',
    'MEM_NV': 'MEM\nNV', 'UP_NV': 'UP\nNV', 'INHIB_NV': 'INHIB\nNV'
}
COLORS = {key: color for key, color in zip(LABELS.keys(), ['#FDB848']*4 + ['#5DADE2']*3 + ['#AF7AC5']*3)}

CLASS_TITLES = {
    'GS': 'Kindergarten', 'CP': '1st grade', 'CE1': '2nd grade', 'CE2': '3rd grade'
}

# Initialize structure to store graph metric
graph_metric = {
    'GRADE': [], 'Degree Centrality': [], 'Strength': [], 'Betweenness': [],
    'Clustering Coefficient': [], 'Modularity': [], 'Global Efficiency': [], 'Local Efficiency': []
}

def build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD):
    G = nx.Graph()
    p_matrix = p_matrices[class_name]  
    
    for task1 in corr_matrix.index:
        for task2 in corr_matrix.columns:
            if task1 != task2:
                corr = corr_matrix.loc[task1, task2]
                p_val = p_matrix.loc[task1, task2]  
                if p_val < significance and corr > 0:
                    G.add_edge(task1, task2, weight=abs(corr))
                    
    return G

def local_efficiency_weighted(G, node):
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return 0
    subgraph = G.subgraph(neighbors)
    efficiency = sum(1 / nx.shortest_path_length(subgraph, u, v, weight='weight') for u, v in combinations(subgraph, 2) if nx.has_path(subgraph, u, v))
    return efficiency * 2 / (len(neighbors) * (len(neighbors) - 1))


def calculate_graph_metric(G):
    degree_centrality = nx.degree_centrality(G)
    strength = dict(G.degree(weight='weight'))
    betweenness = nx.betweenness_centrality(G, weight='weight')
    clustering = nx.clustering(G, weight='weight')
    partition = community_louvain.best_partition(G, weight='weight')
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    modularity = community_louvain.modularity(partition, G)
    global_efficiency = nx.global_efficiency(G)
    local_efficiency = {node: local_efficiency_weighted(G, node) for node in G.nodes}
    
    return {
        'Degree Centrality': degree_centrality, 'Strength': strength,
        'Betweenness': betweenness, 'Clustering Coefficient': clustering,
        'Modularity': modularity, 'Global Efficiency': global_efficiency,
        'Local Efficiency': local_efficiency,
        'Communities': list(communities.values())  
    }

def visualize_graph_with_disconnected(G, class_name, save_path):
    plt.figure(figsize=(6, 6))  
    connected_nodes = set(G.nodes)
    all_nodes = set(LABELS.keys())
    disconnected_nodes = all_nodes - connected_nodes
    pos = nx.spring_layout(G, seed=1, k=3 / np.sqrt(len(G.nodes)), iterations=100)
    x_offset = max(x for x, y in pos.values())  
    y_positions = np.linspace(0.8, 1, len(disconnected_nodes))  
    disconnected_pos = {node: (x_offset, y) for node, y in zip(disconnected_nodes, y_positions)}

    node_sizes_connected = [G.degree(n, weight='weight') * 1200 for n in connected_nodes]
    node_sizes_disconnected = [600] * len(disconnected_nodes) 
    node_colors_connected = [COLORS.get(n, 'gray') for n in connected_nodes]
    node_colors_disconnected = [COLORS.get(n, 'gray') for n in disconnected_nodes]

    nx.draw_networkx_edges(G, pos, width=[d['weight'] * 10 for u, v, d in G.edges(data=True)],
                           edge_color='#383838', alpha=0.6, connectionstyle="arc3,rad=0.2", arrows = True)
    nx.draw_networkx_nodes(G, pos, nodelist=list(connected_nodes),
                           node_color=node_colors_connected, node_size=node_sizes_connected)
    nx.draw_networkx_nodes(disconnected_pos, disconnected_pos, nodelist=list(disconnected_nodes),
                           node_color=node_colors_disconnected, node_size=node_sizes_disconnected, alpha = 0.5)
    nx.draw_networkx_labels(G, pos, labels={n: LABELS[n] for n in connected_nodes}, font_size=8, font_weight='bold')
    nx.draw_networkx_labels(disconnected_pos, pos=disconnected_pos, labels={n: LABELS[n] for n in disconnected_nodes}, font_size=8, font_weight='bold')

    plt.title(CLASS_TITLES.get(class_name, class_name), fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


for class_name, corr_matrix in correlations.items():
    G = build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD)
    if G.number_of_edges() > 0:
        save_path = f"Graph_{class_name}.png"
        visualize_graph_with_disconnected(G, class_name, save_path)
    else:
        print(f"No significant connection for GRADE {class_name}.")

#%% --- Community identification ----
def visualize_graph(G, class_name, communities, save_path):
    plt.figure(figsize=(5, 5))  
    pos = nx.spring_layout(G, seed=1, k=3 / np.sqrt(len(G.nodes)), iterations=100)
    min_distance = 0.2  
    for node1 in pos:
        for node2 in pos:
            if node1 != node2:
                dist = np.linalg.norm(np.array(pos[node1]) - np.array(pos[node2]))
                if dist < min_distance:
                    pos[node2] += (pos[node2] - pos[node1]) * (min_distance - dist)

    node_sizes = [G.degree(n, weight='weight') * 1200 for n in G.nodes]  
    node_colors = [COLORS.get(n, 'gray') for n in G.nodes]
    edge_widths = [d['weight'] * 10 for u, v, d in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='#383838', alpha=0.6,
                           connectionstyle="arc3,rad=0.2", arrows=True)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels={n: LABELS[n] for n in G.nodes}, font_size=8, font_weight='bold')
    
    plt.title(CLASS_TITLES.get(class_name, class_name), fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

#%% --- Graph Metrics ---

def save_metric_to_excel(metric_df, path):
    metric_df.to_excel(path, sheet_name='metric Summary', index=False)
    print(f"Graph metric saved to Excel: {path}")


for class_name, corr_matrix in correlations.items():
    G = build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD)  # Passer class_name pour accéder à p_matrix correct
    if G.number_of_edges() > 0:
        metric = calculate_graph_metric(G)
        
        # Store metric for each class
        graph_metric['GRADE'].append(class_name)
        graph_metric['Degree Centrality'].append(metric['Degree Centrality'])
        graph_metric['Strength'].append(metric['Strength'])
        graph_metric['Clustering Coefficient'].append(metric['Clustering Coefficient'])
        graph_metric['Modularity'].append(metric['Modularity'])
        graph_metric['Global Efficiency'].append(metric['Global Efficiency'])
        save_path = f"Graph_{class_name}.png"
        visualize_graph(G, class_name, metric['Communities'], save_path)

metric_data = [
    {**{'Class': class_name, 'Task': task}, 
     **{metric: graph_metric[metric][idx].get(task, 0) if isinstance(graph_metric[metric][idx], dict) else graph_metric[metric][idx] 
        for metric in ['Degree Centrality', 'Strength', 'Clustering Coefficient', 'Modularity', 'Global Efficiency',]}}
    for idx, class_name in enumerate(graph_metric['GRADE']) for task in LABELS.keys()
]
metric_df = pd.DataFrame(metric_data)
output_excel_path = 'graph_metric.xlsx'
metric_df.to_excel(output_excel_path, sheet_name='Metric Summary', index=False)

print(f"Graph metrics saved to: {output_excel_path}")

            
# %% --- BOOTSTRAP ---

N_BOOTSTRAP = 1000  

bootstrap_results = {
    'GRADE': [],
    'Metric': [],
    'Iteration': [],
    'Task': [],
    'Value': []
}

def bootstrap_graph_metric(data, class_name, tasks, control_variable, n_bootstrap=N_BOOTSTRAP):
    data = data[data['GRADE'] == class_name]

    for i in range(n_bootstrap):
        sample_data = data.sample(frac=1, replace=True)
        correlations = calculate_partial_correlations(sample_data, tasks, control_variable)
        corr_matrix, p_matrix = build_matrices_from_correlations(correlations, tasks)

        G = build_graph(corr_matrix, {'bootstrap': p_matrix}, 'bootstrap', significance=SIGNIFICANCE_THRESHOLD)
        if G.number_of_edges() > 0:
            metric = calculate_graph_metric(G)

            for metric, values in metric.items():
                if metric == 'Communities':
                    continue
                if isinstance(values, dict):
                    for task, task_value in values.items():
                        bootstrap_results['GRADE'].append(class_name)
                        bootstrap_results['Metric'].append(metric)
                        bootstrap_results['Iteration'].append(i + 1)
                        bootstrap_results['Task'].append(task)  
                        bootstrap_results['Value'].append(task_value)
                else:
                    bootstrap_results['GRADE'].append(class_name)
                    bootstrap_results['Metric'].append(metric)
                    bootstrap_results['Iteration'].append(i + 1)
                    bootstrap_results['Task'].append(None) 
                    bootstrap_results['Value'].append(values)

def build_matrices_from_correlations(correlations, tasks):
    corr_matrix = pd.DataFrame(index=tasks, columns=tasks, dtype=float)
    p_matrix = pd.DataFrame(index=tasks, columns=tasks, dtype=float)

    task_pairs, p_values = list(correlations.keys()), [p for _, p in correlations.values()]
    corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

    for idx, (task1, task2) in enumerate(task_pairs):
        corr_matrix.loc[task1, task2] = corr_matrix.loc[task2, task1] = correlations[(task1, task2)][0]
        p_matrix.loc[task1, task2] = p_matrix.loc[task2, task1] = corrected_p_values[idx]

    return corr_matrix, p_matrix

for class_name in data['GRADE'].unique():
    print(f"\nBootstrap for Class {class_name}:")
    bootstrap_graph_metric(data, class_name, tasks, 'MEAN_PARENTS_STUDIES')

bootstrap_df = pd.DataFrame(bootstrap_results)

output_path = 'bootstrap_graph_metric_all_iterations.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for metric in bootstrap_df['Metric'].unique():
        metric_df = bootstrap_df[bootstrap_df['Metric'] == metric][['GRADE', 'Task', 'Iteration', 'Value']]
        metric_df.to_excel(writer, sheet_name=metric, index=False) 

print(f"Bootstrap metric saved to Excel at: {output_path}")

# %%
