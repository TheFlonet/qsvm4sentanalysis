import networkx as nx
from minorminer import find_embedding
from sklearn.metrics import mean_squared_error
from minor_embedding.architectures import toy_svm
from minor_embedding.minors import generate_pegasus
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


def generate_embedding_graph(source: nx.Graph, target: nx.Graph, seed: int) -> nx.Graph | None:
    embedding_i, is_valid = find_embedding(S=source, T=target, random_seed=seed, threads=8, return_overlap=True)
    if is_valid:
        embedding_i_nodes = []
        for nodes in embedding_i.values():
            embedding_i_nodes.extend(nodes)
        return dnx.pegasus_graph(16, node_list=embedding_i_nodes)
    return None


def same_graph_same_location() -> None:
    pegasus = generate_pegasus()
    svm, _ = toy_svm()

    with open('../util/1000.prime') as f:
        primes = [int(x) for x in f.readline().split(', ')]
    print('Seeds loaded, generating embedding...')
    embeddings = [generate_embedding_graph(svm, pegasus, i) for i in primes]

    s = ''
    for i in tqdm(range(len(primes))):
        for j in range(len(primes)):
            if j <= i:
                continue
            if embeddings[i].adj == embeddings[j].adj:
                s += f'Embeddings for {primes[i]} and {primes[j]} are identical\n'

    with open('outputs/same_pos.log', 'w') as f:
        f.write(s)


def same_graph() -> None:
    pegasus = generate_pegasus()
    svm, _ = toy_svm()

    with open('../util/1000.prime') as f:
        primes = [int(x) for x in f.readline().split(', ')]
    print('Seeds loaded, generating embedding...')
    embeddings = [generate_embedding_graph(svm, pegasus, i) for i in primes]

    print('Saving embeddings to disk')
    os.makedirs('embedded_toy_graph', exist_ok=True)
    for i, e in enumerate(embeddings):
        fig = plt.figure()
        dnx.draw_pegasus(e, crosses=True, with_labels=False, ax=fig.add_subplot())
        fig.savefig(f'embedded_toy_graph/{primes[i]}.png', format='png')
        plt.close(fig)
    imgs = [cv2.cvtColor(cv2.imread(f'embedded_toy_graph/{prime}.png'), cv2.COLOR_BGR2GRAY) for prime in primes]

    s = ''
    for i in tqdm(range(len(primes))):
        for j in range(len(primes)):
            if j <= i:
                continue
            mse = mean_squared_error(imgs[i], imgs[j])
            if mse == 0:
                s += f'{primes[i]} - {primes[j]} have the same embedding\n'
    with open('outputs/same_shape.log', 'w') as f:
        f.write(s)

    os.removedirs('embedded_toy_graph')


if __name__ == '__main__':
    # number of nodes in pegasus = 5627
    # same_graph_same_location()
    same_graph()
