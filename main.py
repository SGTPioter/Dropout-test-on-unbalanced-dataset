import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import set_random_seed
from data_loader import get_preprocessed_data


def run_experiment(X_train, y_train, X_val, y_val, rate, dataset_type, num_classes):
    print(f"\nRozpoczęcie treningu: Dropout {rate} na danych '{dataset_type}'")

    clear_session()
    set_random_seed(2026)

    save_dir = os.path.join('models', dataset_type)
    os.makedirs(save_dir, exist_ok=True)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(rate) if rate > 0 else Dropout(0.0),
        Dense(256, activation='relu'),
        Dropout(rate) if rate > 0 else Dropout(0.0),
        Dense(256, activation='relu'),
        Dropout(rate) if rate > 0 else Dropout(0.0),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model_path = os.path.join(save_dir, f'model_dropout_{rate}_best.keras')
    history_path = os.path.join(save_dir, f'model_dropout_{rate}_history.csv')

    callbacks = [
        CSVLogger(history_path),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=500,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Zakończono. Model i historia zapisane w {save_dir}")



scenarios = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

for dataset_type in scenarios:
    X_t, y_t, X_v, y_v, X_te, y_te, num_classes = get_preprocessed_data(dataset_type)

    for rate in rates:
        run_experiment(X_t, y_t, X_v, y_v, rate, dataset_type, num_classes)