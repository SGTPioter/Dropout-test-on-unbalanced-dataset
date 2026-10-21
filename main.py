import os
from pathlib import Path
from data_loader import get_preprocessed_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import set_random_seed


def run_experiment(X_train, y_train, X_val, y_val, dropout_rate):

    run_name = f"model_dropout_{dropout_rate}"
    print(f"\n--- Rozpoczęcie treningu: {run_name} ---")

    # Reset środowiska Keras i stałe ziarno dla powtarzalności (wymóg z lab)
    clear_session()
    set_random_seed(2026)

    # Budowa wielowarstwowej sieci neuronowej
    # Zastosowano inicjalizację HeNormal dla warstw ukrytych (zgodnie z wykładem)
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation="relu", kernel_initializer=HeNormal(seed=2026)),
        Dropout(dropout_rate, seed=2126) if dropout_rate > 0 else Dropout(0.0),

        Dense(256, activation="relu", kernel_initializer=HeNormal(seed=2027)),
        Dropout(dropout_rate, seed=2127) if dropout_rate > 0 else Dropout(0.0),

        Dense(256, activation="relu", kernel_initializer=HeNormal(seed=2028)),
        Dropout(dropout_rate, seed=2128) if dropout_rate > 0 else Dropout(0.0),

        # Warstwa wyjściowa: 7 klas, softmax, inicjalizacja GlorotUniform
        Dense(5, activation="softmax", kernel_initializer=GlorotUniform(seed=2029))
    ])

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    # Przygotowanie folderu i Callbacków (zapis najlepszych wag i historii do CSV)
    Path("models").mkdir(exist_ok=True)

    callbacks = [
        CSVLogger(f"models/{run_name}_history.csv"),
        ModelCheckpoint(
            filepath=f"models/{run_name}_best.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,  # Wczesne zatrzymanie zapobiegające marnowaniu zasobów
            restore_best_weights=True  # Zawsze przywracamy najlepsze wagi
        )
    ]

    # Trening
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,  # Odpowiednio dużo, by model Vanilla zdążył się przeuczyć
        batch_size=500,  # Zwiększony batch size (Covtype to potężny zbiór, 32 trwałoby za długo na CPU)
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    print(f"--- Zakończono: {run_name} ---")



print("Ładowanie i przetwarzanie danych...")
X_t, y_t, X_v, y_v, X_te, y_te = get_preprocessed_data()

# Zgodnie z treścią projektu: porównanie min. 3 konfiguracji, zmieniając JEDEN czynnik
# Wybrany czynnik: Prawdopodobieństwo Dropout

for rate in (0.0, 0.2, 0.5):
    run_experiment(X_t, y_t, X_v, y_v, rate)

print("\nSkończone! Modele i logi CSV zapisane w folderze 'models/'.")