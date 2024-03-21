from dataset.dataset_creation import *
from dataset.embeddings_generator import *


def main():
    splits = ('train', 'test')
    seed = 7
    dataset = load_tweet_eval()
    dataset = to_binary(dataset, splits, seed)
    dataset = normalize(dataset, splits, seed)
    dataset = add_sentence_embedding(dataset, splits)
    save(dataset, splits, './data/')


if __name__ == '__main__':
    main()
