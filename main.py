from sklearn.svm import SVC

from classical.CSVM import CSVM
from dataset.dataset_creation import *
from dataset.embeddings_generator import *
from sklearn.model_selection import train_test_split
from util.visualization import plot_dataset, plot_decision_boundary
from util.evaluation import evaluate

DEBUG = True


def main_debug():
    print('Generating dataset'.upper())
    examples, labels = get_dummy_dataset()
    plot_dataset(examples, labels)
    examples_train, examples_test, labels_train, labels_test = train_test_split(examples, labels,
                                                                                test_size=0.4, random_state=7)
    for method_name, svm_generator in (('sklearn', SVC), ('cplex', CSVM), ('d-wave', None)):
        svm_model = svm_generator(kernel='rbf', C=1.0, gamma=0.1)
        print(f'Training with {method_name}'.upper())
        svm_model.fit(examples_train, labels_train)
        plot_decision_boundary(svm_model, examples_train, labels_train)
        predictions = svm_model.predict(examples_test)
        print(f'Testing with {method_name}'.upper())
        evaluate(labels_test, predictions)
        print('-' * 50)


def main():
    if DEBUG:
        main_debug()
    else:
        splits = ('train', 'test')
        seed = 7
        dataset = load_tweet_eval()
        dataset = to_binary(dataset, splits, seed)
        dataset = normalize(dataset, splits, seed)
        dataset = add_sentence_embedding(dataset, splits)
        save(dataset, splits, './data/')


if __name__ == '__main__':
    main()
