import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv


def load_mutual_information_pairs(path: str):
    mutual_information_pairs = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            mutual_information_pairs.append((line['P1'], line['P2'], float(line['mf'])))
    return mutual_information_pairs


if __name__=='__main__':
    # The following analysis is more suited to be in a Notebook in ad hoc analysis
    path = '/Users/Aaron/Desktop/mf_labHpo_labHpo - mf_labHpo_labHpo.csv'
    mf_paris = load_mutual_information_pairs(path)
    mf_pairs_filters = [x for x in mf_paris if x[2] >= 0.20]
    G = nx.Graph()
    G.add_weighted_edges_from(mf_pairs_filters)

    partition = community_louvain.best_partition(G)
    print(partition)

    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    labels = {x : x for x in partition.keys()}
    nx.draw_networkx(G, pos, nodelist=partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()), labels=labels, font_size=7)

    plt.show()





