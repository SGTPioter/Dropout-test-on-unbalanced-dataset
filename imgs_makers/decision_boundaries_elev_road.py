import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import get_preprocessed_data

# Dynamiczne ustalenie głównego folderu projektu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

F1_IDX = 0  # Elevation
F2_IDX = 5  # Horizontal Dist to Roadways

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
fig.suptitle('Granice decyzyjne modeli (Elevation vs Horizontal Dist to Roadways)',
             fontsize=20, fontweight='bold', y=0.96)

colors_hex = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#313695']
cmap_background = ListedColormap(colors_hex)
cmap_points = ListedColormap(colors_hex)

for col_idx, dataset in enumerate(datasets):
    X_t, y_t, X_v, y_v, X_te, y_te, num_classes = get_preprocessed_data(dataset)

    # Próbkowanie do wizualizacji - 5000 punktów, aby nie przeciążyć wykresu
    sample_size = min(5000, len(X_te))
    X_sample, _, y_sample, _ = train_test_split(
        X_te, y_te, train_size=sample_size, stratify=y_te, random_state=2026
    )
    y_sample = y_sample.flatten()

    mean_features = np.mean(X_sample, axis=0)

    # Ustalenie zakresów dla siatki (grid) na podstawie wybranych dwóch cech
    x_min, x_max = X_sample[:, F1_IDX].min() - 0.5, X_sample[:, F1_IDX].max() + 0.5
    y_min, y_max = X_sample[:, F2_IDX].min() - 0.5, X_sample[:, F2_IDX].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))

    # Wypełnianie reszty z 54 cech wartościami średnimi dla siatki tła
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    grid_data = np.tile(mean_features, (mesh_points.shape[0], 1))
    grid_data[:, F1_IDX] = mesh_points[:, 0]
    grid_data[:, F2_IDX] = mesh_points[:, 1]

    for row_idx, rate in enumerate(rates):
        ax = axes[row_idx, col_idx]
        model_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_best.keras')

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)

            # Predykcje wyłącznie dla siatki tła
            Z_probs = model.predict(grid_data, batch_size=4096, verbose=0)
            Z = np.argmax(Z_probs, axis=1)
            Z = Z.reshape(xx.shape)

            # Rysowanie tła (obszarów decyzyjnych)
            ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_background, levels=np.arange(-0.5, 7.5, 1))

            # Rysowanie rzeczywistych punktów na wierzchu
            ax.scatter(X_sample[:, F1_IDX], X_sample[:, F2_IDX], c=y_sample,
                       cmap=cmap_points, edgecolor='black', s=20, alpha=0.9, vmin=0, vmax=6)

            # Czysty tytuł, bez skali Micro F1
            ax.set_title(f'{dataset.upper()} | Dropout: {rate}', fontsize=14, pad=10)

        ax.set_xlabel('Elevation (Z-score)')
        if col_idx == 0:
            ax.set_ylabel('Horizontal Dist to Roadways (Z-score)')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'decision_boundaries_elev_road_clean.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Wygenerowano czysty wykres granic decyzyjnych i zapisano w: {save_path}")