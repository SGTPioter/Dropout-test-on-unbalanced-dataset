import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from data_loader import get_preprocessed_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))
fig.suptitle('Kształt oznaczonych granic decyzyjnych w rzucie PCA', fontsize=20, fontweight='bold', y=0.96)

colors_hex = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#313695']
cmap_background = ListedColormap(colors_hex)
cmap_points = ListedColormap(colors_hex)

for col_idx, dataset in enumerate(datasets):
    X_t, y_t, X_v, y_v, X_te, y_te, num_classes = get_preprocessed_data(dataset)

    sample_size = min(5000, len(X_te))
    X_sample, _, y_sample, _ = train_test_split(
        X_te, y_te, train_size=sample_size, stratify=y_te, random_state=2026
    )
    y_sample = y_sample.flatten()

    pca = PCA(n_components=2, random_state=2026)
    X_sample_pca = pca.fit_transform(X_sample)

    x_min, x_max = X_sample_pca[:, 0].min() - 1, X_sample_pca[:, 0].max() + 1
    y_min, y_max = X_sample_pca[:, 1].min() - 1, X_sample_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))

    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_54D = pca.inverse_transform(grid_pca)

    for row_idx, rate in enumerate(rates):
        ax = axes[row_idx, col_idx]
        model_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_best.keras')

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            Z_probs = model.predict(grid_54D, batch_size=4096, verbose=0)

            # Twarda decyzja (argmax) zamiast mieszania kolorów
            Z = np.argmax(Z_probs, axis=1)
            Z = Z.reshape(xx.shape)

            # Rysowanie ostrych granic (contourf)
            ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_background, levels=np.arange(-0.5, 7.5, 1))
            ax.scatter(X_sample_pca[:, 0], X_sample_pca[:, 1], c=y_sample,
                       cmap=cmap_points, edgecolor='black', s=20, alpha=0.9, vmin=0, vmax=6)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_title('')

        if row_idx == 0:
            ax.set_title(dataset.upper(), fontsize=16, pad=10)

        if col_idx == 0:
            ax.set_ylabel(f'Dropout: {rate}', fontsize=14, labelpad=10)
        else:
            ax.set_ylabel('')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
script_name = os.path.basename(__file__)
image_name = os.path.splitext(script_name)[0] + '.png'
save_path = os.path.join(save_dir, image_name)

plt.savefig(save_path, dpi=300, bbox_inches='tight')