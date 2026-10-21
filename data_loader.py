import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_preprocessed_data(data_home="./data", random_state=67):

    covtype = fetch_covtype()
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

        # Jeśli to klasa 0 lub 1 i przekracza dozwolony limit, losujemy próbkę
        if cls in [0, 1] and len(cls_indices) > max_allowed:
            np.random.seed(random_state)
            sampled_indices = np.random.choice(cls_indices, max_allowed, replace=False)
            indices_to_keep.extend(sampled_indices)
        else:
            indices_to_keep.extend(cls_indices)

        # Odfiltrowanie zbalansowanego zbioru
    features = features[indices_to_keep]
    labels = labels[indices_to_keep]


    X_train, X_hold, y_train, y_hold = train_test_split(
        features, labels, test_size=0.30, stratify=labels, random_state=random_state
    )

    # Podział Holdout na Val 50% / Test 50% (czyli po 15% całości)
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, stratify=y_hold, random_state=random_state
    )

    # Standaryzacja wyłącznie pierwszych 10 cech (zapobieganie Data Leakage)
    scaler = StandardScaler()
    X_train[:, :10] = scaler.fit_transform(X_train[:, :10])
    X_val[:, :10] = scaler.transform(X_val[:, :10])
    X_test[:, :10] = scaler.transform(X_test[:, :10])

    return X_train, y_train, X_val, y_val, X_test, y_test



X_t, y_t, X_v, y_v, X_te, y_te = get_preprocessed_data()
print(f"Train: {X_t.shape[0]} | Val: {X_v.shape[0]} | Test: {X_te.shape[0]}")
print(f"Średnia cechy 0 (Train) po skalowaniu: {X_t[:, 0].mean():.4f}")
print(f"Odchylenie cechy 0 (Train) po skalowaniu: {X_t[:, 0].std():.4f}")