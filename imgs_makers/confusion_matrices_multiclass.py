import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score
from data_loader import get_preprocessed_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 20))
fig.suptitle('Wieloklasowe macierze pomyłek z ewaluacją (Strata i Czułość)', fontsize=20, fontweight='bold', y=0.96)

for col_idx, dataset in enumerate(datasets):
    X_t, y_t, X_v, y_v, X_te, y_te, num_classes = get_preprocessed_data(dataset)
    y_te_flat = y_te.flatten()

    for row_idx, rate in enumerate(rates):
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
                    ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontweight='bold', fontsize=12)

            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))
            ax.set_xticklabels([f'Kl. {i}' for i in range(num_classes)])
            ax.set_yticklabels([f'Kl. {i}' for i in range(num_classes)])

            ax.set_title(
                f'Zbiór: {dataset.upper()} | Dropout: {rate}\nStrata: {loss:.3f} | Czułość (Macro): {recall:.3f}',
                fontsize=14, pad=10)
            ax.set_xlabel('Przewidywana klasa')

            if col_idx == 0:
                ax.set_ylabel('Rzeczywista klasa')
            else:
                ax.set_ylabel('')
        else:
            ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
script_name = os.path.basename(__file__)
image_name = os.path.splitext(script_name)[0] + '.png'
save_path = os.path.join(save_dir, image_name)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Zapisano w: {save_path}")