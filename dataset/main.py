from dataset.dataset_creation import load_tweet_eval, to_binary, normalize, save
from dataset.embeddings_generator import add_sentence_embedding


def generate_data(seed: int) -> None:
    splits = ('train', 'test')
    dataset = load_tweet_eval()
    dataset = to_binary(dataset, splits, seed)
    dataset = normalize(dataset, splits, seed)
    dataset = add_sentence_embedding(dataset, splits)
    save(dataset, splits, '../data/')


if __name__ == '__main__':
    generate_data(7)
