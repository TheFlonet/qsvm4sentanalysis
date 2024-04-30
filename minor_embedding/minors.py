import time
import networkx as nx
from dotenv import load_dotenv
from dwave.system import DWaveSampler
from matplotlib import pyplot as plt
import dwave_networkx as dnx
from minorminer import find_embedding


def generate_pegasus() -> nx.Graph:
    load_dotenv()
    qpu = DWaveSampler()
    qpu_edges = qpu.edgelist
    qpu_nodes = qpu.nodelist

    return dnx.pegasus_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)


def search_embedding(source: nx.Graph, target: nx.Graph, out_path: str, seed: int=7):
    s = time.time()
    embedding, is_valid = find_embedding(S=source, T=target, random_seed=seed, threads=8, return_overlap=True)
    if is_valid:
        fig = plt.figure()
        dnx.draw_pegasus_embedding(target, embedding, node_size=0.1, ax=fig.add_subplot(),
                                   width=0.05, crosses=True, linewidths=0)
        fig.savefig(out_path, format='svg')
        plt.close(fig)
        print(f'saved solution -', end=' ')
    else:
        print(f'not a valid solution -', end=' ')
    print(time.time() - s)
