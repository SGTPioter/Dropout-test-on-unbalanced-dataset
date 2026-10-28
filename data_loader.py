import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DROP = [3, 4] # usuwane klasy (numeracja CLEAR)
LUT = np.full(7, -1, dtype=np.int64)
LUT[[0, 1, 2, 5, 6]] = np.arange(5) # CLEAR -> EDITED: 0,1,2,5,6 -> 0..4


def split_clear(random_state=67):
    """JEDEN podział na pełnym zbiorze; surowe (nieskalowane) cechy."""
    cov = fetch_covtype()
    X = cov.data.astype(np.float32)
    y = (cov.target - 1).astype(np.int64)
    X_tr, X_h, y_tr, y_h = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state)
    X_va, X_te, y_va, y_te = train_test_split(
        X_h, y_h, test_size=0.50, stratify=y_h, random_state=random_state)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def edit_split(X, y, random_state=67, factor=3):
    """Edycja jednego podzbioru: wycięcie klas, remap, ograniczenie klas 0 i 1."""
    keep = ~np.isin(y, DROP)
    X, y = X[keep], LUT[y[keep]]

    cap = np.bincount(y).min() * factor
    rng = np.random.default_rng(random_state)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        if c in (0, 1) and len(ci) > cap:
            ci = rng.choice(ci, cap, replace=False)
        idx.append(ci)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def fit_scaler(X_train):
    return StandardScaler().fit(X_train[:, :10])


def apply_scaler(scaler, X):
    X = X.copy()
    X[:, :10] = scaler.transform(X[:, :10])
    return X


def get_preprocessed_data(dataset_type, random_state=67, return_scaler=False):
    tr, va, te = split_clear(random_state)
    if dataset_type == 'edited':
        tr, va, te = (edit_split(*s, random_state) for s in (tr, va, te))
        num_classes = 5
    else:
        num_classes = 7

    scaler = fit_scaler(tr[0])   # tylko na treningu danego wariantu
    X_tr, X_va, X_te = (apply_scaler(scaler, s[0]) for s in (tr, va, te))
    print(f"\n--- {dataset_type.upper()} --- Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    out = (X_tr, tr[1], X_va, va[1], X_te, te[1], num_classes)
    return out + (scaler,) if return_scaler else out