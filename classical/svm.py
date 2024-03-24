from sklearn.svm import SVC


def get_sklearn_svm() -> SVC:
    return SVC(kernel='rbf', C=1.0, gamma='scale')
