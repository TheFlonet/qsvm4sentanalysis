import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from classical.CSVM import CSVM
from quantum.QSVM import QSVM
import matplotlib
import plotly.graph_objects as go


def plot_decision_boundary(model: SVC | CSVM | QSVM, examples: np.ndarray, labels: np.ndarray, title: str) -> None:
    h = 0.02
    x_min, x_max = examples[:, 0].min() - 1, examples[:, 0].max() + 1
    y_min, y_max = examples[:, 1].min() - 1, examples[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=matplotlib.colormaps['coolwarm'], alpha=0.8)
    plt.scatter(examples[:, 0], examples[:, 1], c=labels, cmap=matplotlib.colormaps['coolwarm'], edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundary')
    plt.savefig(f'img/decision_boundary_{title}.png')
    # plt.show()
    plt.clf()


def plot_generalized_boundary(model: SVC | CSVM | QSVM, examples: np.ndarray, labels: np.ndarray, title: str) -> None:
    pca = PCA(n_components=3)
    reduced_examples = pca.fit_transform(examples)
    model.fit(reduced_examples, labels)

    h = 0.1
    x_min, x_max = reduced_examples[:, 0].min() - 1, reduced_examples[:, 0].max() + 1
    y_min, y_max = reduced_examples[:, 1].min() - 1, reduced_examples[:, 1].max() + 1
    z_min, z_max = reduced_examples[:, 2].min() - 1, reduced_examples[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = model.predict(grid_points).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=reduced_examples[:, 0],
        y=reduced_examples[:, 1],
        z=reduced_examples[:, 2],
        mode='markers',
        marker=dict(size=5, color=labels, colorscale='bluered', opacity=0.8),
        name='3D decision boundary'
    ))

    for i, z_slice in enumerate(np.arange(z_min, z_max, h)):
        fig.add_trace(go.Surface(
            x=xx[:, :, i],
            y=yy[:, :, i],
            z=np.full_like(xx[:, :, i], z_slice),
            surfacecolor=Z[:, :, i],
            colorscale='bluered',
            showscale=False,
            opacity=0.05
        ))

    fig.update_layout(
        title=f'3D Decision Boundary - {title}',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_dataset(examples: np.ndarray, labels: np.ndarray) -> None:
    plt.scatter(examples[:, 0], examples[:, 1], c=labels)
    plt.savefig('img/dataset.png')
    # plt.show()
    plt.clf()
