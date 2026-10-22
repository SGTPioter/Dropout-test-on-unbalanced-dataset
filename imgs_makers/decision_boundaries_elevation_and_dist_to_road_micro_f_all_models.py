import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from data_loader import get_preprocessed_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

F1_IDX = 0
F2_IDX = 5

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
fig.suptitle('Granice decyzyjne modeli (Elevation vs Horizontal Dist to Roadways)',
             fontsize=20, fontweight='bold', y=0.96)

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
    y_te_flat = y_te.flatten()

    mean_features = np.mean(X_sample, axis=0)

    x_min, x_max = X_sample[:, F1_IDX].min() - 0.5, X_sample[:, F1_IDX].max() + 0.5
    y_min, y_max = X_sample[:, F2_IDX].min() - 0.5, X_sample[:, F2_IDX].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    grid_data = np.tile(mean_features, (mesh_points.shape[0], 1))
    grid_data[:, F1_IDX] = mesh_points[:, 0]
    grid_data[:, F2_IDX] = mesh_points[:, 1]

    for row_idx, rate in enumerate(rates):
        ax = axes[row_idx, col_idx]
        model_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_best.keras')

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)

            y_pred_probs = model.predict(X_te, batch_size=4096, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            micro_f1 = f1_score(y_te_flat, y_pred, average='micro')

            Z_probs = model.predict(grid_data, batch_size=4096, verbose=0)
            Z = np.argmax(Z_probs, axis=1)
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_background, levels=np.arange(-0.5, 7.5, 1))

            ax.scatter(X_sample[:, F1_IDX], X_sample[:, F2_IDX], c=y_sample,
                       cmap=cmap_points, edgecolor='black', s=20, alpha=0.9, vmin=0, vmax=6)

            ax.set_title(f'{dataset.upper()} | Dropout: {rate} | Micro F1: {micro_f1:.4f}', fontsize=14, pad=10)

        ax.set_xlabel('Elevation (Z-score)')

        if col_idx == 0:
            ax.set_ylabel('Horizontal Dist to Roadways (Z-score)')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'granice_decyzyjne_elev_road.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')