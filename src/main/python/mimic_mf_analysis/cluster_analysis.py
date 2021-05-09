import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv
from obonetx.ontology import Ontology
import re
import pandas as pd


def load_mutual_information_pairs(path: str):
    mutual_information_pairs = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            mutual_information_pairs.append(('P1_' + line['P1'], 'P2_' + line['P2'], float(line['mf'])))
    return mutual_information_pairs


if __name__=='__main__':
    hpo = Ontology('/Users/Aaron/git/human-phenotype-ontology/hp.obo')
    hpo_term_map = hpo.term_id_2_label_map()

    # The following analysis is more suited to be in a Notebook in ad hoc analysis
    path = '/Users/Aaron/Desktop/mf_textHpo_labHpo - mf_textHpo_labHpo.csv'
    mf_paris = load_mutual_information_pairs(path)
    mf_pairs_filters = [x for x in mf_paris if x[2] >= 0.07]
    G = nx.Graph()
    G.add_weighted_edges_from(mf_pairs_filters)

    partition = community_louvain.best_partition(G)

    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    labels = {}
    labels_textHpo = {x : 'textHpo: ' + hpo_term_map.get(re.sub('P1_', '', x), x) for x in partition.keys() if x.startswith('P1_')}
    labels_labHpo = {x: 'labHpo: ' + hpo_term_map.get(re.sub('P2_', '', x), x) for x in partition.keys() if
                     x.startswith('P2_')}
    labels.update(labels_textHpo)
    labels.update(labels_labHpo)
    print(pd.DataFrame(data={'node': partition.keys(), 'partition': partition.values()}))
    # print([1 if x.startswith('textHpo') else 2 for x in labels.values()])
    # labels = {x : hpo_term_map.get(re.sub('P[12]_', '', x), x) for x in partition.keys()}
    # nx.draw_networkx(G, pos, nodelist=partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()), labels=labels, font_size=7)
    nx.draw_networkx(G, pos, nodelist=partition.keys(), node_size=40, cmap=cmap, node_color=[1 if x.startswith('P1') else 3 for x in partition.keys()],
                     labels=labels, font_size=7)

    plt.show()





