import numpy as np
import os
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def get_preprocessed_data(data_home="./data", random_state=67):
    print("Pobieranie i balansowanie zbioru danych (tylko do odczytu)...")
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

    return X_val, y_val


def plot_macro_f1_bar_chart(model_path='models/model_dropout_0.2_best.keras'):
    X_val, y_val = get_preprocessed_data()
    class_names = ['Spruce/Fir', 'Lodgepole', 'Ponderosa', 'Douglas-fir', 'Krummholz']

    print(f"Wczytywanie wag z zapisanego pliku: {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Błąd wczytywania modelu. Upewnij się, że plik istnieje. Szczegóły: {e}")
        return

    print("Obliczanie predykcji...")
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    f1_per_class = f1_score(y_val, y_pred, average=None) * 100
    f1_macro = f1_score(y_val, y_pred, average='macro') * 100

    print("Generowanie wykresu...")
    fig, ax = plt.subplots(figsize=(10, 6))

    min_idx = np.argmin(f1_per_class)
    colors = ['#117a8b' if i != min_idx else '#d98324' for i in range(len(class_names))]

    bars = ax.bar(class_names, f1_per_class, color=colors, edgecolor='none')

    ax.axhline(y=f1_macro, color='#1e528b', linestyle='--', linewidth=2, zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_ylim(0, 105)

    ax.set_ylabel('F1 [%]', fontsize=12, labelpad=10)
    ax.set_xlabel('Nazwa klasy', fontsize=12, labelpad=10)

    plt.title('Makro-F1: każda klasa ma równą wagę', fontsize=18, fontweight='bold', loc='left', pad=20)

    formula_text = f"F1_makro = {f1_macro:.1f}%\n\nSłabszy wynik ({class_names[min_idx]})\nwpływa na średnią tak samo\njak wynik każdej innej klasy."
    plt.figtext(0.55, 0.5, formula_text, fontsize=12, va='center')

    os.makedirs('../imgs', exist_ok=True)
    output_img = os.path.join('../imgs', 'macro_f1_bar_chart.png')

    plt.tight_layout()
    plt.subplots_adjust(right=0.5)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Gotowe! Zapisano wykres jako '{output_img}'.")
    plt.show()



plot_macro_f1_bar_chart(model_path='../models/edited/model_dropout_0.2_best.keras')