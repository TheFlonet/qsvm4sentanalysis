import random
import os
from dataset.create_datasets import create_datasets, load_datasets


def main():
    if not os.listdir('data'):
        create_datasets()
    train, test, validation = load_datasets()
    print(train, test, validation)


if __name__ == '__main__':
    random.seed(58481)
    main()
