import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score
from data_loader import get_preprocessed_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

# indeks klasy w modelu EDITED -> oryginalny indeks klasy w CLEAR
edited_to_clear_mapping = np.array([0, 1, 2, 5, 6])
excluded_classes = [3, 4]

# Dane CLEAR (wspólny zbiór testowy) wraz ze scalerem CLEAR
_, _, _, _, X_te_clear, y_te_clear, num_classes_clear, scaler_clear = get_preprocessed_data(
    'clear', return_scaler=True)
# Scaler EDITED (dopasowany na zbiorze treningowym EDITED, na którym uczono modele EDITED)
*_, scaler_edited = get_preprocessed_data('edited', return_scaler=True)
y_te_flat = y_te_clear.flatten()

# Test CLEAR w skali EDITED: cofnięcie skalowania CLEAR i nałożenie skalowania EDITED (pierwsze 10 cech).
X_te_for_edited = X_te_clear.copy()
X_te_for_edited[:, :10] = scaler_edited.transform(scaler_clear.inverse_transform(X_te_clear[:, :10]))

# Zmiana układu na 2 wiersze i 3 kolumny: wiersz 0 -> CLEAR, wiersz 1 -> EDITED
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

for row_idx, dataset in enumerate(datasets):
    X_input = X_te_clear if dataset == 'clear' else X_te_for_edited

    for col_idx, rate in enumerate(rates):
        ax = axes[row_idx, col_idx]
        model_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_best.keras')

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)

            if dataset == 'clear':
                loss, _ = model.evaluate(X_input, y_te_clear, verbose=0)
                loss_str = f"{loss:.3f}"
            else:
                loss_str = "N/A"

            y_pred_probs = model.predict(X_input, batch_size=4096, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            if dataset == 'edited':
                y_pred = edited_to_clear_mapping[y_pred]

            recalls_per_class = recall_score(y_te_flat, y_pred, labels=np.arange(num_classes_clear),
                                             average=None, zero_division=0)

            if dataset == 'clear':
                recall = np.mean(recalls_per_class) # średnia z wszystkich 7 klas dla CLEAR
            else:
                valid_classes = [i for i in range(num_classes_clear) if i not in excluded_classes]
                recall = np.mean(recalls_per_class[valid_classes])  # średnia z 5 klas dla EDITED

            cm_raw = confusion_matrix(y_te_flat, y_pred, labels=np.arange(num_classes_clear))

            row_sums = cm_raw.sum(axis=1)
            row_sums[row_sums == 0] = 1
            cm_norm = cm_raw.astype('float') / row_sums[:, np.newaxis]

            ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)

            if dataset == 'edited':
                for ex_c in excluded_classes:
                    ax.add_patch(
                        patches.Rectangle((-0.5, ex_c - 0.5), num_classes_clear, 1, fill=True, color='lightgray',
                                          alpha=0.9, zorder=1))
                    ax.add_patch(
                        patches.Rectangle((ex_c - 0.5, -0.5), 1, num_classes_clear, fill=True, color='lightgray',
                                          alpha=0.9, zorder=1))

            for i in range(num_classes_clear):
                for j in range(num_classes_clear):
                    val = cm_raw[i, j]
                    is_excluded = dataset == 'edited' and (i in excluded_classes or j in excluded_classes)

                    if is_excluded:
                        text_color = "gray" if val == 0 else "black"
                    else:
                        text_color = "white" if cm_norm[i, j] > 0.5 else "black"

                    ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontweight='bold', fontsize=11,
                            zorder=2)

            ax.set_xticks(np.arange(num_classes_clear))
            ax.set_yticks(np.arange(num_classes_clear))
            ax.set_xticklabels([f'Kl. {i}' for i in range(num_classes_clear)])
            ax.set_yticklabels([f'Kl. {i}' for i in range(num_classes_clear)])

            recall_label = "Czułość (macro, 7 klas)" if dataset == 'clear' else "Czułość (macro, 5 klas)"
            ax.set_title(
                f'Model: {dataset.upper()} | Dropout: {rate}\nStrata: {loss_str} | {recall_label}: {recall:.3f}',
                fontsize=13, pad=10
            )

            if row_idx == 1:
                ax.set_xlabel('Przewidywana klasa', fontsize=11)
            else:
                ax.set_xlabel('')

            if col_idx == 0:
                ax.set_ylabel('Rzeczywista klasa', fontsize=11)
            else:
                ax.set_ylabel('')
        else:
            ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.90)

save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'confusion_matrices_unified_test.png')

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Zapisano w: {save_path}")