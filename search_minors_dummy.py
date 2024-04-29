from minor_embedding.architectures import svm_dummy_dataset
from minor_embedding.minors import generate_pegasus, search_embedding


def main():
    pegasus = generate_pegasus()
    svm = svm_dummy_dataset()

    with open('util/1000.prime') as f:
        primes = [int(x) for x in f.readline().split(', ')][:100]

    for index, i in enumerate(primes):
        print(f'{index}/{len(primes)}: {i} -', end=' ')
        search_embedding(svm, pegasus, f'./minors_dummy/embedding_seed_{str(i).zfill(4)}.svg', i)


if __name__ == '__main__':
    main()
