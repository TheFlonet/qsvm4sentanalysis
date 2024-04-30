import networkx as nx
from minorminer import find_embedding
from minor_embedding.architectures import toy_svm
from minor_embedding.minors import generate_pegasus
import dwave_networkx as dnx


def generate_embedding_graph(source: nx.Graph, target: nx.Graph, seed: int) -> nx.Graph:
    embedding_i = find_embedding(S=source, T=target, random_seed=seed, threads=8)
    embedding_i_nodes = []
    for nodes in embedding_i.values():
        embedding_i_nodes.extend(nodes)
    return dnx.pegasus_graph(16, node_list=embedding_i_nodes)


def main() -> None:
    pegasus = generate_pegasus()
    svm = toy_svm()

    with open('../util/1000.prime') as f:
        primes = [int(x) for x in f.readline().split(', ')]
    print('Seeds loaded, generating embedding...')
    embeddings = [generate_embedding_graph(svm, pegasus, i) for i in primes]

    for i in range(len(primes)):
        if i % 100 == 0:
            print(f'{i}/{len(primes)}')
        for j in range(len(primes)):
            if j <= i:
                continue
            if embeddings[i].adj == embeddings[j].adj:
                print(f'Embeddings for {primes[i]} and {primes[j]} are identical')

    '''
    Results:
    
    Embeddings for 19 and 6451 are identical
    Embeddings for 107 and 4871 are identical
    Embeddings for 419 and 6947 are identical
    Embeddings for 859 and 2789 are identical
    Embeddings for 967 and 7523 are identical
    Embeddings for 1237 and 6967 are identical
    Embeddings for 1699 and 6703 are identical
    Embeddings for 1709 and 4789 are identical
    Embeddings for 1979 and 5107 are identical
    Embeddings for 2389 and 3761 are identical
    Embeddings for 2917 and 3041 are identical
    Embeddings for 3571 and 3793 are identical
    Embeddings for 3877 and 6709 are identical
    Embeddings for 4027 and 6199 are identical
    Embeddings for 4813 and 7789 are identical
    Embeddings for 5179 and 6737 are identical
    '''


if __name__ == '__main__':
    main()
