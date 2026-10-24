import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from data_loader import get_preprocessed_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
fig.suptitle('Raport klasyfikacji z sklearn (Precyzja, Czułość, F1) dla każdej klasy',
             fontsize=20, fontweight='bold', y=0.96)

for col_idx, dataset in enumerate(datasets):
    X_t, y_t, X_v, y_v, X_te, y_te, num_classes = get_preprocessed_data(dataset)
    y_te_flat = y_te.flatten()

    for row_idx, rate in enumerate(rates):
        ax = axes[row_idx, col_idx]
        model_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_best.keras')

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)

            y_pred_probs = model.predict(X_te, batch_size=4096, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            report = classification_report(y_te_flat, y_pred, output_dict=True, zero_division=0)

            metrics_data = np.zeros((num_classes, 3))
            for c in range(num_classes):
                c_str = str(c)
                if c_str in report:
                    metrics_data[c, 0] = report[c_str]['precision']
                    metrics_data[c, 1] = report[c_str]['recall']
                    metrics_data[c, 2] = report[c_str]['f1-score']

            im = ax.imshow(metrics_data, cmap='Blues', vmin=0.0, vmax=1.0, aspect='auto')

            for i in range(num_classes):
                for j in range(3):
                    val = metrics_data[i, j]
                    text_color = "black" if 0.25 < val < 0.75 else "white"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight='bold',
                            fontsize=12)

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Precyzja (Precision)', 'Czułość (Recall)', 'F1-Score'], fontsize=12)
            ax.set_yticks(np.arange(num_classes))
            ax.set_yticklabels([f'Klasa {i}' for i in range(num_classes)], fontsize=12)

            ax.set_title(f'{dataset.upper()} | Dropout: {rate}', fontsize=16, pad=10)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
script_name = os.path.basename(__file__)
image_name = os.path.splitext(script_name)[0] + '.png'
save_path = os.path.join(save_dir, image_name)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Zapisano w: {save_path}")