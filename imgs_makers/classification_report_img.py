import numpy as np
import os
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def get_preprocessed_data(data_home="./data", random_state=67):
    print("Pobieranie i balansowanie zbioru danych...")
    covtype = fetch_covtype(data_home=data_home)
    features = covtype.data.astype(np.float32)
    labels = (covtype.target - 1).astype(np.int64).reshape(-1, 1)

    mask_keep = ~np.isin(labels.flatten(), [3, 4])
    features = features[mask_keep]
    labels = labels[mask_keep]

    mapping = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}
    labels = np.vectorize(mapping.get)(labels)

    unique, counts = np.unique(labels, return_counts=True)
    min_class_count = np.min(counts)
    max_allowed = min_class_count * 3

    indices_to_keep = []
    for cls in unique:
        cls_indices = np.where(labels.flatten() == cls)[0]
        if cls in [0, 1] and len(cls_indices) > max_allowed:
            np.random.seed(random_state)
            sampled_indices = np.random.choice(cls_indices, max_allowed, replace=False)
            indices_to_keep.extend(sampled_indices)
        else:
            indices_to_keep.extend(cls_indices)

    np.random.seed(random_state)
    np.random.shuffle(indices_to_keep)

    features = features[indices_to_keep]
    labels = labels[indices_to_keep]

    X_train, X_hold, y_train, y_hold = train_test_split(
        features, labels, test_size=0.30, stratify=labels, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, stratify=y_hold, random_state=random_state
    )

    scaler = StandardScaler()
    X_train[:, :10] = scaler.fit_transform(X_train[:, :10])
    X_val[:, :10] = scaler.transform(X_val[:, :10])
    X_test[:, :10] = scaler.transform(X_test[:, :10])

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # 1. Załadowanie zbioru walidacyjnego
    _, _, X_val, y_val, _, _ = get_preprocessed_data()
    new_class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Douglas-fir', 'Krummholz']

    # 2. Definicja siatki 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Analiza Modeli: Macierze Pomyłek i Metryki Ewaluacyjne", fontsize=18, fontweight='bold', y=0.98)

    positions = {
        0.2: axes[0, 0],
        0.5: axes[0, 1],
        0.0: axes[1, 0]
    }

    metrics_summary = []

    # 3. Pętla rysująca macierze i licząca statystyki
    for rate, ax in positions.items():
        model_path = os.path.join('models', f'model_dropout_{rate}_best.keras')
        print(f"Przetwarzanie modelu: {model_path}...")

        try:
            model = load_model(model_path)
            y_pred_prob = model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

            metrics_summary.append([
                f"Dropout {rate}",
                f"{acc * 100:.1f}%",
                f"{prec * 100:.1f}%",
                f"{rec * 100:.1f}%",
                f"{f1 * 100:.1f}%"
            ])

            cm_normalized = confusion_matrix(y_val, y_pred, normalize='true')
            sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues',
                        xticklabels=new_class_names, yticklabels=new_class_names,
                        ax=ax, cbar=False)

            ax.set_title(f'Dropout: {rate}', fontsize=15, pad=10)
            ax.set_ylabel('Prawdziwa klasa')
            ax.set_xlabel('Przewidziana klasa')
            ax.tick_params(axis='x', rotation=45)

        except Exception as e:
            print(f"BŁĄD przy wariancie {rate}: {e}")
            ax.set_title(f'Dropout: {rate} (Brak pliku)', fontsize=15)
            ax.axis('off')

    # --- NOWE: Sortowanie wierszy tabeli alfabetycznie (0.0 -> 0.2 -> 0.5) ---
    metrics_summary.sort(key=lambda x: x[0])

    # 4. Zagospodarowanie czwartego pola (Tabela)
    ax_table = axes[1, 1]
    ax_table.axis('off')

    col_labels = ['Model', 'Trafność\n(Accuracy)', 'Precyzja\n(Precision)', 'Pełność\n(Recall)', 'F-miara\n(F1-Score)']

    table = ax_table.table(
        cellText=metrics_summary,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.25, 0.9, 0.5]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#dcefff')
        cell.set_edgecolor('black')

    ax_table.set_title('Zbiorcze Zestawienie Metryk (Średnia Makro)', fontsize=15, pad=10)

    # 5. Zapis do pliku
    plt.tight_layout()

    # --- ZMIANY: Utworzenie folderu i zapis do imgs/ ---
    os.makedirs('imgs', exist_ok=True)
    output_img = os.path.join('imgs', 'confusion_matrices_grid_2x2.png')

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\nSukces! Zapisano układ ze spersonalizowaną tabelą jako '{output_img}'.")
    plt.show()