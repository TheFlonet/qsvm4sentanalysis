from minor_embedding.architectures import svm_dummy_dataset, toy_svm
from minor_embedding.minors import generate_pegasus, search_embedding


def main() -> None:
    pegasus = generate_pegasus()
    # svm = svm_dummy_dataset()
    svm = toy_svm()

    with open('./util/1000.prime') as f:
        primes = [int(x) for x in f.readline().split(', ')]

    for index, i in enumerate(primes):
        print(f'{index}/{len(primes)}: {i} -', end=' ')
        search_embedding(svm, pegasus, f'./minor_embedding/minors_toy/embedding_seed_{str(i).zfill(4)}.svg', i)


if __name__ == '__main__':
    main()
