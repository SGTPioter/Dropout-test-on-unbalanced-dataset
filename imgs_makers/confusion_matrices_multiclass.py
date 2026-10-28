import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score
from data_loader import get_preprocessed_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

# Układ 2 wiersze na 3 kolumny: wiersz 0 -> CLEAR (3 dropouty), wiersz 1 -> EDITED (3 dropouty)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

for row_idx, dataset in enumerate(datasets):
    X_t, y_t, X_v, y_v, X_te, y_te, num_classes = get_preprocessed_data(dataset)
    y_te_flat = y_te.flatten()

    for col_idx, rate in enumerate(rates):
        ax = axes[row_idx, col_idx]
        model_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_best.keras')

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)

            loss, _ = model.evaluate(X_te, y_te, verbose=0)

            y_pred_probs = model.predict(X_te, batch_size=4096, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            recall = recall_score(y_te_flat, y_pred, average='macro', zero_division=0)

            cm_raw = confusion_matrix(y_te_flat, y_pred, labels=np.arange(num_classes))

            row_sums = cm_raw.sum(axis=1)
            row_sums[row_sums == 0] = 1
            cm_norm = cm_raw.astype('float') / row_sums[:, np.newaxis]

            ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)

            for i in range(num_classes):
                for j in range(num_classes):
                    val = cm_raw[i, j]
                    text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontweight='bold', fontsize=11)

            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))

            if dataset == 'edited':
                ax.set_xticklabels(['Kl. 0', 'Kl. 1', 'Kl. 2', 'Kl. 5', 'Kl. 6'])
                ax.set_yticklabels(['Kl. 0', 'Kl. 1', 'Kl. 2', 'Kl. 5', 'Kl. 6'])
            else:
                ax.set_xticklabels([f'Kl. {i}' for i in range(num_classes)])
                ax.set_yticklabels([f'Kl. {i}' for i in range(num_classes)])

            ax.set_title(
                f'Zbiór: {dataset.upper()} | Dropout: {rate}\nStrata: {loss:.3f} | Czułość (Macro): {recall:.3f}',
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
save_path = os.path.join(save_dir, 'confusion_matrices_multiclass.png')

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Zapisano w: {save_path}")