import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses, metrics
from tensorflow.keras.utils import set_random_seed


def plot_normal_triangle_boundaries():
    os.makedirs('plots', exist_ok=True)

    print("Ładowanie i próbkowanie danych Covertype...")
    covtype = fetch_covtype(data_home="./data")
    labels_all = (covtype.target - 1).astype(np.int64)

    # Próbkowanie 5000 punktów ze stratyfikacją
    _, X_sample, _, y_sample = train_test_split(
        covtype.data[:, [0, 3]], labels_all,
        test_size=5000, stratify=labels_all, random_state=67
    )

    X_sample = X_sample.astype(np.float32)
    y_sample = y_sample.reshape(-1, 1)

    scaler = StandardScaler()
    X_sample = scaler.fit_transform(X_sample)

    X_train, X_val, y_train, y_val = train_test_split(
        X_sample, y_sample, test_size=0.3, stratify=y_sample, random_state=67
    )

    colors_bg = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc', '#ffccff', '#ccffff', '#e0e0e0']
    colors_pt = ['#ff0000', '#00cc00', '#0000ff', '#cccc00', '#cc00cc', '#00cccc', '#707070']
    class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow',
                   'Aspen', 'Douglas-fir', 'Krummholz']
    cmap_background = ListedColormap(colors_bg)

    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    dropout_rates = [0.0, 0.2, 0.5]

    # Tworzenie płótna i siatki 2 wiersze x 4 kolumny
    fig = plt.figure(figsize=(14, 11), dpi=150)
    gs = gridspec.GridSpec(2, 4, figure=fig)

    # ZMIANA UKŁADU: Klasyczny trójkąt
    ax1 = fig.add_subplot(gs[0, 1:3])  # Górny środkowy (kolumny 1-2)
    ax2 = fig.add_subplot(gs[1, 0:2])  # Lewy dolny (kolumny 0-1)
    ax3 = fig.add_subplot(gs[1, 2:4])  # Prawy dolny (kolumny 2-3)
    axes = [ax1, ax2, ax3]

    fig.suptitle("Porównanie granic decyzyjnych w zależności od parametru Dropout",
                 fontsize=18, fontweight='bold', y=0.96)

    for idx, (rate, ax) in enumerate(zip(dropout_rates, axes)):
        set_random_seed(67)
        print(f"Trenowanie modelu (Dropout: {rate})...")

        model = Sequential([
            Input(shape=(2,)),
            Dense(128, activation="relu"),
            Dropout(rate) if rate > 0 else Dropout(0.0),
            Dense(128, activation="relu"),
            Dropout(rate) if rate > 0 else Dropout(0.0),
            Dense(64, activation="relu"),
            Dropout(rate) if rate > 0 else Dropout(0.0),
            Dense(7, activation="softmax")
        ])

        model.compile(optimizer=Adam(learning_rate=0.01),
                      loss=losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=60, batch_size=128, verbose=0)
        val_acc = history.history['val_accuracy'][-1] * 100

        preds_prob = model.predict(grid_points, batch_size=2048, verbose=0)
        Z = np.argmax(preds_prob, axis=1).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=np.arange(8) - 0.5, cmap=cmap_background, alpha=0.7)
        ax.contour(xx, yy, Z, levels=np.arange(7) - 0.5, colors='black', linewidths=0.5, alpha=0.5)

        for i in range(7):
            mask = (y_train.flatten() == i)
            ax.scatter(X_train[mask, 0], X_train[mask, 1], c=colors_pt[i],
                       edgecolor='k', s=15, alpha=0.8)

        ax.set_title(f"Dropout: {rate} (Val Acc: {val_acc:.1f}%)", fontsize=14, pad=10)
        ax.set_xlabel("Wysokość n.p.m.")
        ax.set_ylabel("Odległość do wody")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Tworzenie legendy
    handles, labels = [], []
    for i in range(7):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_pt[i],
                                  markeredgecolor='k', markersize=8))
        labels.append(f"Klasa {i}: {class_names[i]}")

    # Umieszczenie legendy w prawym górnym rogu obok wykresu Dropout 0.0
    ax_legend = fig.add_subplot(gs[0, 3])
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', title="Gatunki drzew",
                     fontsize=11, title_fontsize=13, frameon=True)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.08, wspace=0.3, hspace=0.35)

    output_path = 'plots/normal_triangle_decision_boundaries.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nZapisano zbiorczy wykres: '{output_path}'")


if __name__ == "__main__":
    plot_normal_triangle_boundaries()