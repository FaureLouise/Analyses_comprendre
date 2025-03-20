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
import random 
#%%
# Chemin du fichier Excel
chemin_fichier_excel = "C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/DATA/DATA_IES_Z_nov_24.xlsx"
data = pd.read_excel(chemin_fichier_excel)

taches = ['DP', 'DL', 'SL', 'CS', 'MAJ_V', 'MAJ_NV', 'MDT_V', 'MDT_NV', 'INHIB_V', 'INHIB_NV']
#%%
"""Création des matrices de corrélations partielles"""
# Configuration de base
tasks = ['DP', 'DL', 'SL', 'CS', 'MDT_V', 'MAJ_V', 'INHIB_V', 'MDT_NV', 'MAJ_NV', 'INHIB_NV']
labels = {'DP': 'PHO', 'DL': 'LEX-R', 'SL': 'VOC', 'CS': 'SYN', 'MDT_V': 'MEM V', 'MAJ_V': 'UP V', 
          'INHIB_V': 'INHIB V', 'MDT_NV': 'MEM NV', 'MAJ_NV': 'UP NV', 'INHIB_NV': 'INHIB NV'}
colors = {'DP': '#E68E00', 'DL': '#E68E00', 'SL': '#E68E00', 'CS': '#E68E00', 
          'MDT_V': '#0080D5', 'MAJ_V': '#0080D5', 'INHIB_V': '#0080D5',
          'MDT_NV': '#9000CB', 'MAJ_NV': '#9000CB', 'INHIB_NV': '#9000CB'}
class_titles = {'GS': 'Kindergarten', 'CP': '1st grade', 'CE1': '2nd grade', 'CE2': '3rd grade'}

# Fonction pour calculer toutes les corrélations partielles d'une classe donnée
def calculate_partial_correlations(data, tasks, control_variable):
    correlations = {}
    for i, task1 in enumerate(tasks):
        for task2 in tasks[i+1:]:
            result = pg.partial_corr(data=data, x=task1, y=task2, covar=[control_variable], method='spearman')
            correlations[(task1, task2)] = (result['r'].values[0], result['p-val'].values[0])
    return correlations

# Fonction pour générer les heatmaps
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

    save_path = f"C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/Figures/heatmap_corr_p_{cls}.png"

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                significance = "*" if p_val < 0.001 else "*" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                # Couleur et style du texte selon les conditions
                if corr_val < 0 or p_val >= 0.05:  # Corrélations négatives ou non significatives
                    plt.text(j, i, f"{corr_val:.2f}\n{significance}", 
                             ha='center', va='center', color='#575757', fontsize=14, fontweight='normal')
                else:  # Corrélations positives et significatives
                    plt.text(j, i, f"{corr_val:.2f}\n{significance}", 
                             ha='center', va='center', color='black', fontsize=14, fontweight='bold')


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Calcul des corrélations partielles et affichage des heatmaps pour chaque classe
def process_all_classes(data, tasks, control_variable):
    correlations = {}
    p_matrices = {}  # Define p_matrices here

    for cls in data['CLASSE'].unique():
        print(f"\nProcessing Class {cls}:")
        class_data = data[data['CLASSE'] == cls]
        class_correlations = calculate_partial_correlations(class_data, tasks, control_variable)

        corr_matrix = pd.DataFrame(index=tasks, columns=tasks, dtype=float)
        p_matrix = pd.DataFrame(index=tasks, columns=tasks, dtype=float)

        # Fill correlation and p-value matrices with FDR-corrected p-values
        task_pairs, p_values = list(class_correlations.keys()), [p for _, p in class_correlations.values()]
        corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

        for idx, (task1, task2) in enumerate(task_pairs):
            corr_matrix.loc[task1, task2] = corr_matrix.loc[task2, task1] = class_correlations[(task1, task2)][0]
            p_matrix.loc[task1, task2] = p_matrix.loc[task2, task1] = corrected_p_values[idx]

        correlations[cls] = corr_matrix
        p_matrices[cls] = p_matrix  # Store the p-matrix for the current class

        # Optional: Plot heatmap
        save_path = f"heatmap_corr_partielle_{cls}.png"
        plot_heatmap(corr_matrix, p_matrix, cls, tasks, labels, colors, class_titles, save_path)

    return correlations, p_matrices  # Return both dictionaries

# Appel principal
# Main call to process all classes and get correlations and p-values matrices
correlations, p_matrices = process_all_classes(data, tasks, 'MOY_ETUDE_PARENT')
#%% """Affichage textuel des corrélations avec significativité pour chaque classe'''
# Configuration de base

# Affichage textuel des corrélations avec significativité pour chaque classe
def print_textual_correlations(data, tasks, control_variable):
    for cls in data['CLASSE'].unique():
        print(f"\nResults for Class {class_titles.get(cls, cls)}:")
        class_data = data[data['CLASSE'] == cls]
        class_correlations = calculate_partial_correlations(class_data, tasks, control_variable)

        # Extraction des p-valeurs pour correction FDR
        task_pairs, p_values = list(class_correlations.keys()), [p for _, p in class_correlations.values()]
        corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

        # Impression des résultats avec corrélation et significativité
        for idx, (task_pair, (corr_val, _)) in enumerate(class_correlations.items()):
            task1, task2 = task_pair
            corrected_p_val = corrected_p_values[idx]
            significance = "***" if corrected_p_val < 0.001 else "**" if corrected_p_val < 0.01 else "*" if corrected_p_val < 0.05 else ""
            print(f"Correlation between {task1} and {task2}: r = {corr_val:.2f}, p = {corrected_p_val:.4f} {significance}")

# Appel principal pour afficher les corrélations textuelles
print_textual_correlations(data, tasks, 'MOY_ETUDE_PARENT')
# %% '''GRAPHE INITIAL - bug local efficiency ?
SIGNIFICANCE_THRESHOLD = 0.05
# Mapping for node labels
LABELS = {
    'DP': 'PHO', 'DL': 'LEX-R', 'SL': 'VOC', 'CS': 'SYN',
    'MDT_V': 'MEM\nV', 'MAJ_V': 'UP\nV', 'INHIB_V': 'INHIB\nV',
    'MDT_NV': 'MEM\nNV', 'MAJ_NV': 'UP\nNV', 'INHIB_NV': 'INHIB\nNV'
}
COLORS = {key: color for key, color in zip(LABELS.keys(), ['#FDB848']*4 + ['#5DADE2']*3 + ['#AF7AC5']*3)}

CLASS_TITLES = {
    'GS': 'Kindergarten', 'CP': '1st grade', 'CE1': '2nd grade', 'CE2': '3rd grade'
}

# Initialize structure to store graph metrics
graph_metrics = {
    'Classe': [], 'Degree Centrality': [], 'Strength': [], 'Betweenness': [],
    'Clustering Coefficient': [], 'Modularity': [], 'Global Efficiency': [], 'Local Efficiency': []
}

def build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD):
    G = nx.Graph()
    p_matrix = p_matrices[class_name]  # Accéder à la matrice p pour cette classe spécifique
    
    for task1 in corr_matrix.index:
        for task2 in corr_matrix.columns:
            if task1 != task2:
                corr = corr_matrix.loc[task1, task2]
                p_val = p_matrix.loc[task1, task2]  # Accéder aux p-valeurs pour cette paire de tâches
                # Ajouter uniquement si la corrélation est positive et significative
                if p_val < significance and corr > 0:
                    G.add_edge(task1, task2, weight=abs(corr))
                    
    return G


# Calcul de l'efficacité locale pondérée
def local_efficiency_weighted(G, node):
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return 0
    subgraph = G.subgraph(neighbors)
    efficiency = sum(1 / nx.shortest_path_length(subgraph, u, v, weight='weight') for u, v in combinations(subgraph, 2) if nx.has_path(subgraph, u, v))
    return efficiency * 2 / (len(neighbors) * (len(neighbors) - 1))

 # Import de la bibliothèque pour l'algorithme de Louvain

def calculate_graph_metrics(G):
    """Calculates various metrics on the graph G using Louvain for community detection."""
    degree_centrality = nx.degree_centrality(G)
    strength = dict(G.degree(weight='weight'))
    betweenness = nx.betweenness_centrality(G, weight='weight')
    clustering = nx.clustering(G, weight='weight')
    
    # Utilisation de l'algorithme de Louvain pour détecter les communautés
    partition = community_louvain.best_partition(G, weight='weight')
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    # Calcul de la modularité basée sur la partition détectée par Louvain
    modularity = community_louvain.modularity(partition, G)
    global_efficiency = nx.global_efficiency(G)
    local_efficiency = {node: local_efficiency_weighted(G, node) for node in G.nodes}
    
    return {
        'Degree Centrality': degree_centrality, 'Strength': strength,
        'Betweenness': betweenness, 'Clustering Coefficient': clustering,
        'Modularity': modularity, 'Global Efficiency': global_efficiency,
        'Local Efficiency': local_efficiency,
        'Communities': list(communities.values())  # Liste de communautés pour la visualisation
    }

def visualize_graph_with_disconnected(G, class_name, save_path):
    plt.figure(figsize=(6, 6))  # Taille ajustée pour inclure les nœuds déconnectés

    # Identifier les nœuds connectés et déconnectés
    connected_nodes = set(G.nodes)
    all_nodes = set(LABELS.keys())
    disconnected_nodes = all_nodes - connected_nodes

    # Utiliser spring_layout pour la disposition des nœuds connectés
    pos = nx.spring_layout(G, seed=1, k=3 / np.sqrt(len(G.nodes)), iterations=100)

    # Positionner les nœuds déconnectés à la verticale
    x_offset = max(x for x, y in pos.values())  # Décalage horizontal vers la droite
    y_positions = np.linspace(0.8, 1, len(disconnected_nodes))  # Espacement vertical uniforme
    disconnected_pos = {node: (x_offset, y) for node, y in zip(disconnected_nodes, y_positions)}

    # Fusionner les positions des nœuds connectés et déconnectés
    full_pos = {**pos, **disconnected_pos}

    # Définir les tailles et couleurs des nœuds
    node_sizes_connected = [G.degree(n, weight='weight') * 1200 for n in connected_nodes]
    node_sizes_disconnected = [600] * len(disconnected_nodes)  # Taille fixe pour les nœuds déconnectés
    node_colors_connected = [COLORS.get(n, 'gray') for n in connected_nodes]
    node_colors_disconnected = [COLORS.get(n, 'gray') for n in disconnected_nodes]

    # Dessiner les arêtes avec courbure pour les nœuds connectés
    nx.draw_networkx_edges(G, pos, width=[d['weight'] * 10 for u, v, d in G.edges(data=True)],
                           edge_color='#383838', alpha=0.6, connectionstyle="arc3,rad=0.2", arrows = True)

    # Dessiner les nœuds connectés
    nx.draw_networkx_nodes(G, pos, nodelist=list(connected_nodes),
                           node_color=node_colors_connected, node_size=node_sizes_connected)

    # Dessiner les nœuds déconnectés
    nx.draw_networkx_nodes(disconnected_pos, disconnected_pos, nodelist=list(disconnected_nodes),
                           node_color=node_colors_disconnected, node_size=node_sizes_disconnected, alpha = 0.5)

    # Ajouter les étiquettes pour les nœuds connectés
    nx.draw_networkx_labels(G, pos, labels={n: LABELS[n] for n in connected_nodes}, font_size=8, font_weight='bold')

    # Ajouter les étiquettes pour les nœuds déconnectés
    nx.draw_networkx_labels(disconnected_pos, pos=disconnected_pos, labels={n: LABELS[n] for n in disconnected_nodes}, font_size=8, font_weight='bold')

    # Ajouter le titre de la classe
    plt.title(CLASS_TITLES.get(class_name, class_name), fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


for class_name, corr_matrix in correlations.items():
    # Construire le graphe pour la classe donnée
    G = build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD)
    
    # Vérifier si le graphe contient des arêtes
    if G.number_of_edges() > 0:
        # Visualiser le graphe avec les nœuds déconnectés
        save_path = f"C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/Figures/Graph_{class_name}.png"
        visualize_graph_with_disconnected(G, class_name, save_path)
    else:
        print(f"Aucune connexion significative pour la classe {class_name}.")

#%%
def visualize_graph(G, class_name, communities, save_path):
    """Visualizes the graph with node colors based on Louvain communities, and custom edge styles."""
    plt.figure(figsize=(5, 5))  # Augmentation de la taille de la figure pour plus d'espace

    # Utiliser spring_layout avec une forte répulsion pour minimiser le chevauchement
    pos = nx.spring_layout(G, seed=1, k=3 / np.sqrt(len(G.nodes)), iterations=100)

    # Appliquer une distance minimale entre les nœuds pour éviter le chevauchement
    min_distance = 0.2  # Distance minimale souhaitée
    for node1 in pos:
        for node2 in pos:
            if node1 != node2:
                dist = np.linalg.norm(np.array(pos[node1]) - np.array(pos[node2]))
                # Si la distance est trop petite, ajuster les positions
                if dist < min_distance:
                    # Décale légèrement la position pour éviter le chevauchement
                    pos[node2] += (pos[node2] - pos[node1]) * (min_distance - dist)

    # Définir les tailles et couleurs des nœuds
    node_sizes = [G.degree(n, weight='weight') * 1200 for n in G.nodes]  # Taille ajustée pour éviter chevauchement
    node_colors = [COLORS.get(n, 'gray') for n in G.nodes]
    edge_widths = [d['weight'] * 10 for u, v, d in G.edges(data=True)]

    # Dessiner les arêtes avec courbure pour une meilleure lisibilité
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='#383838', alpha=0.6,
                           connectionstyle="arc3,rad=0.2", arrows=True)

    # Dessiner les nœuds et les étiquettes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels={n: LABELS[n] for n in G.nodes}, font_size=8, font_weight='bold')
    
    # Ajouter le titre de la classe
    plt.title(CLASS_TITLES.get(class_name, class_name), fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# Enregistrement des métriques dans un fichier Excel
def save_metrics_to_excel(metrics_df, path):
    metrics_df.to_excel(path, sheet_name='Metrics Summary', index=False)
    print(f"Graph metrics saved to Excel: {path}")

# Main loop for each class
# Main loop for each class
for class_name, corr_matrix in correlations.items():
    G = build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD)  # Passer class_name pour accéder à p_matrix correct
    if G.number_of_edges() > 0:
        metrics = calculate_graph_metrics(G)
        
        # Store metrics for each class
        graph_metrics['Classe'].append(class_name)
        graph_metrics['Degree Centrality'].append(metrics['Degree Centrality'])
        graph_metrics['Strength'].append(metrics['Strength'])
        graph_metrics['Betweenness'].append(metrics['Betweenness'])
        graph_metrics['Clustering Coefficient'].append(metrics['Clustering Coefficient'])
        graph_metrics['Modularity'].append(metrics['Modularity'])
        graph_metrics['Global Efficiency'].append(metrics['Global Efficiency'])
        graph_metrics['Local Efficiency'].append(metrics['Local Efficiency'])

        # Visualize and save graph
        save_path = f"C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/Figures/Graph_{class_name}.png"
        visualize_graph(G, class_name, metrics['Communities'], save_path)

metrics_data = [
    {**{'Class': class_name, 'Task': task}, 
     **{metric: graph_metrics[metric][idx].get(task, 0) if isinstance(graph_metrics[metric][idx], dict) else graph_metrics[metric][idx] 
        for metric in ['Degree Centrality', 'Strength', 'Betweenness', 'Clustering Coefficient', 'Modularity', 'Global Efficiency', 'Local Efficiency']}}
    for idx, class_name in enumerate(graph_metrics['Classe']) for task in LABELS.keys()
]
metrics_df = pd.DataFrame(metrics_data)
output_excel_path = 'C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/graph_metrics.xlsx'
#save_metrics_to_excel(metrics_df, output_excel_path)    
            
#%% ''' local efficiency sans modification
import networkx as nx
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from community import community_louvain

SIGNIFICANCE_THRESHOLD = 0.05

# Mapping for node labels
LABELS = {
    'DP': 'PHO', 'DL': 'LEX-D', 'SL': 'VOC', 'CS': 'SYN',
    'MDT_V': 'MEM\nV', 'MAJ_V': 'UP\nV', 'INHIB_V': 'INHIB\nV',
    'MDT_NV': 'MEM\nNV', 'MAJ_NV': 'UP\nNV', 'INHIB_NV': 'INHIB\nNV'
}
COLORS = {key: color for key, color in zip(LABELS.keys(), ['#FDB848']*4 + ['#5DADE2']*3 + ['#AF7AC5']*3)}

CLASS_TITLES = {
    'GS': 'Kindergarten', 'CP': '1st grade', 'CE1': '2nd grade', 'CE2': '3rd grade'
}

# Initialize structure to store graph metrics
graph_metrics = {
    'Classe': [], 'Degree Centrality': [], 'Strength': [], 'Betweenness': [],
    'Clustering Coefficient': [], 'Modularity': [], 'Global Efficiency': [], 'Local Efficiency': []
}

def build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD):
    G = nx.Graph()
    p_matrix = p_matrices[class_name]  # Accéder à la matrice p pour cette classe spécifique
    
    for task1 in corr_matrix.index:
        for task2 in corr_matrix.columns:
            if task1 != task2:
                corr = corr_matrix.loc[task1, task2]
                p_val = p_matrix.loc[task1, task2]  # Accéder aux p-valeurs pour cette paire de tâches
                # Ajouter uniquement si la corrélation est positive et significative
                if p_val < significance and corr > 0:
                    G.add_edge(task1, task2, weight=corr)                 
    return G

def calculate_graph_metrics(G):
    """Calculates various metrics on the graph G using Louvain for community detection."""
    degree_centrality = nx.degree_centrality(G)
    strength = dict(G.degree(weight='weight'))
    betweenness = nx.betweenness_centrality(G, weight='weight')
    clustering = nx.clustering(G, weight='weight')
    
    # Utilisation de l'algorithme de Louvain pour détecter les communautés
    partition = community_louvain.best_partition(G, weight='weight')
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    # Calcul de la modularité basée sur la partition détectée par Louvain
    modularity = community_louvain.modularity(partition, G)
    global_efficiency = nx.global_efficiency(G)
    
    # Calcul de l'efficacité locale pour chaque nœud
    local_efficiency_per_node = {
        node: nx.local_efficiency(G.subgraph(G.neighbors(node)))
        for node in G.nodes()
    }
    
    return {
        'Degree Centrality': degree_centrality, 'Strength': strength,
        'Betweenness': betweenness,
        'Clustering Coefficient': clustering,
        'Modularity': modularity,
        'Global Efficiency': global_efficiency,
        'Local Efficiency': local_efficiency_per_node,  # Par nœud
        'Communities': list(communities.values())  # Liste de communautés pour la visualisation
    }

def visualize_graph(G, class_name, communities, save_path):
    """Visualizes the graph with node colors based on Louvain communities, and custom edge styles."""
    plt.figure(figsize=(8, 8))  # Taille ajustée pour éviter le chevauchement des nœuds

    pos = nx.spring_layout(G, seed=1)  # Utilisation de spring_layout

    # Définir les tailles et couleurs des nœuds
    node_sizes = [G.degree(n, weight='weight') * 800 for n in G.nodes]
    node_colors = [COLORS.get(n, 'gray') for n in G.nodes]
    edge_widths = [d['weight'] * 3 for u, v, d in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels={n: LABELS[n] for n in G.nodes}, font_size=10)
    
    plt.title(CLASS_TITLES.get(class_name, class_name), fontsize=16)
    plt.axis('off')
    plt.savefig(save_path, dpi=300)
    plt.show()

# Main loop for each class
for class_name, corr_matrix in correlations.items():
    G = build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD)
    if G.number_of_edges() > 0:
        metrics = calculate_graph_metrics(G)
        
        # Store metrics for each class
        graph_metrics['Classe'].append(class_name)
        graph_metrics['Degree Centrality'].append(metrics['Degree Centrality'])
        graph_metrics['Strength'].append(metrics['Strength'])
        graph_metrics['Betweenness'].append(metrics['Betweenness'])
        graph_metrics['Clustering Coefficient'].append(metrics['Clustering Coefficient'])
        graph_metrics['Modularity'].append(metrics['Modularity'])
        graph_metrics['Global Efficiency'].append(metrics['Global Efficiency'])
        graph_metrics['Local Efficiency'].append(metrics['Local Efficiency'])  # Par nœud

        # Visualize and save graph
        save_path = f"C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/Figures/Graph_{class_name}.png"
        visualize_graph(G, class_name, metrics['Communities'], save_path)

# Convert metrics to a DataFrame
metrics_data = [
    {**{'Class': class_name, 'Task': task}, 
     **{metric: graph_metrics[metric][idx].get(task, 0) if isinstance(graph_metrics[metric][idx], dict) else graph_metrics[metric][idx] 
        for metric in ['Degree Centrality', 'Strength', 'Betweenness', 'Clustering Coefficient', 'Modularity', 'Global Efficiency']},
     'Local Efficiency': graph_metrics['Local Efficiency'][idx].get(task, 0)}  # Par nœud
    for idx, class_name in enumerate(graph_metrics['Classe']) for task in LABELS.keys()
]
metrics_df = pd.DataFrame(metrics_data)

# Save metrics to Excel
output_excel_path = 'C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/graph_metrics.xlsx'
metrics_df.to_excel(output_excel_path, index=False)



# Création d'un exemple de graphe

LABELS = {
    'DP': 'PHO', 'DL': 'LEX-D', 'SL': 'VOC', 'CS': 'SYN',
    'MDT_V': 'MEM\nV', 'MAJ_V': 'UP\nV', 'INHIB_V': 'INHIB\nV',
    'MDT_NV': 'MEM\nNV', 'MAJ_NV': 'UP\nNV', 'INHIB_NV': 'INHIB\nNV'
}

COLORS = {key: color for key, color in zip(LABELS.keys(), ['#FDB848']*4 + ['#5DADE2']*3 + ['#AF7AC5']*3)}

CLASS_TITLES = {
    'GS': 'Kindergarten', 'CP': '1st grade', 'CE1': '2nd grade', 'CE2': '3rd grade'
}

def build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD):
    G = nx.Graph()
    p_matrix = p_matrices[class_name]  # Accéder à la matrice p pour cette classe spécifique
    
    for task1 in corr_matrix.index:
        for task2 in corr_matrix.columns:
            if task1 != task2:
                corr = corr_matrix.loc[task1, task2]
                p_val = p_matrix.loc[task1, task2]  # Accéder aux p-valeurs pour cette paire de tâches
                # Ajouter uniquement si la corrélation est positive et significative
                if p_val < significance and corr > 0:
                    G.add_edge(task1, task2, weight=abs(corr))
    return G

def visualize_graph(G, class_name):
    """Visualizes the graph with node colors based on Louvain communities."""
    plt.figure(figsize=(5, 5))  # Augmentation de la taille de la figure pour plus d'espace

    # Utiliser spring_layout avec une forte répulsion pour minimiser le chevauchement
    pos = nx.spring_layout(G, seed=1, k=3 / np.sqrt(len(G.nodes)), iterations=100)

    # Appliquer une distance minimale entre les nœuds pour éviter le chevauchement
    min_distance = 0.2  # Distance minimale souhaitée
    for node1 in pos:
        for node2 in pos:
            if node1 != node2:
                dist = np.linalg.norm(np.array(pos[node1]) - np.array(pos[node2]))
                # Si la distance est trop petite, ajuster les positions
                if dist < min_distance:
                    # Décale légèrement la position pour éviter le chevauchement
                    pos[node2] += (pos[node2] - pos[node1]) * (min_distance - dist)

    # Appliquer l'algorithme Louvain pour détecter les communautés
    partition = community_louvain.best_partition(G, weight='weight')
    
    # Assigner une couleur différente à chaque communauté
    community_colors = list(set(partition.values()))
    node_colors = [community_colors.index(partition[node]) for node in G.nodes]

    # Dessiner les arêtes avec courbure pour une meilleure lisibilité
    nx.draw_networkx_edges(G, pos, width=1, edge_color='#383838', alpha=0.6, 
                           connectionstyle="arc3,rad=0.2", arrows=True)

    # Dessiner les nœuds avec les couleurs des communautés
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.rainbow, node_size=800)
    nx.draw_networkx_labels(G, pos, labels={n: LABELS.get(n, n) for n in G.nodes}, font_size=8, font_weight='bold')
    
    # Ajouter le titre de la classe
    plt.title(CLASS_TITLES.get(class_name, class_name), fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/Figures/Graph_commu_{class_name}.png", dpi=300)
    plt.show()

# Main loop for each class
for class_name, corr_matrix in correlations.items():
    G = build_graph(corr_matrix, p_matrices, class_name, significance=SIGNIFICANCE_THRESHOLD)
    if G.number_of_edges() > 0:
        visualize_graph(G, class_name)


# %%
# Définition de la méthode de bootstrap pour les métriques de graphes
N_BOOTSTRAP = 1000  # Nombre d'itérations de bootstrap

bootstrap_results = {
    'Classe': [],
    'Metric': [],
    'Iteration': [],
    'Task': [],
    'Value': []
}

def bootstrap_graph_metrics(data, class_name, tasks, control_variable, n_bootstrap=N_BOOTSTRAP):
    """Perform bootstrap on data for a given class and return metrics for each iteration."""
    class_data = data[data['CLASSE'] == class_name]

    for i in range(n_bootstrap):
        sample_data = class_data.sample(frac=1, replace=True)

        # Calculate partial correlations for the bootstrap sample
        correlations = calculate_partial_correlations(sample_data, tasks, control_variable)
        
        # Generate correlation and p-value matrices for the bootstrap sample
        corr_matrix, p_matrix = build_matrices_from_correlations(correlations, tasks)

        # Build the graph for the current bootstrap sample
        G = build_graph(corr_matrix, {'bootstrap': p_matrix}, 'bootstrap', significance=SIGNIFICANCE_THRESHOLD)
        if G.number_of_edges() > 0:
            metrics = calculate_graph_metrics(G)

            for metric, values in metrics.items():
                if metric == 'Communities':
                    continue
                if isinstance(values, dict):
                    for task, task_value in values.items():
                        bootstrap_results['Classe'].append(class_name)
                        bootstrap_results['Metric'].append(metric)
                        bootstrap_results['Iteration'].append(i + 1)
                        bootstrap_results['Task'].append(task)  # Adding specific task here
                        bootstrap_results['Value'].append(task_value)
                else:
                    bootstrap_results['Classe'].append(class_name)
                    bootstrap_results['Metric'].append(metric)
                    bootstrap_results['Iteration'].append(i + 1)
                    bootstrap_results['Task'].append(None)  # Add placeholder for metrics without a task
                    bootstrap_results['Value'].append(values)

def build_matrices_from_correlations(correlations, tasks):
    """Construct correlation and p-value matrices from partial correlation results."""
    corr_matrix = pd.DataFrame(index=tasks, columns=tasks, dtype=float)
    p_matrix = pd.DataFrame(index=tasks, columns=tasks, dtype=float)

    # Extract values and apply FDR correction for p-values
    task_pairs, p_values = list(correlations.keys()), [p for _, p in correlations.values()]
    corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

    for idx, (task1, task2) in enumerate(task_pairs):
        corr_matrix.loc[task1, task2] = corr_matrix.loc[task2, task1] = correlations[(task1, task2)][0]
        p_matrix.loc[task1, task2] = p_matrix.loc[task2, task1] = corrected_p_values[idx]

    return corr_matrix, p_matrix

# Main loop to run bootstrap for each class
for class_name in data['CLASSE'].unique():
    print(f"\nBootstrap for Class {class_name}:")
    bootstrap_graph_metrics(data, class_name, tasks, 'MOY_ETUDE_PARENT')

# Convert bootstrap results to a DataFrame
bootstrap_df = pd.DataFrame(bootstrap_results)

# Save bootstrap results to an Excel file with one sheet per metric
output_path = 'C:/Users/Louise/OneDrive - Université Grenoble Alpes/Bureau/THESE/ARTICLES/ARTICLE MEMOIRE/bootstrap_graph_metrics_all_iterations.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for metric in bootstrap_df['Metric'].unique():
        metric_df = bootstrap_df[bootstrap_df['Metric'] == metric][['Classe', 'Task', 'Iteration', 'Value']]
        metric_df.to_excel(writer, sheet_name=metric, index=False)  # Include Task column here

print(f"Bootstrap metrics saved to Excel at: {output_path}")

# %%
