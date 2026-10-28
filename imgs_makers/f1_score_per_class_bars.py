import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from data_loader import get_preprocessed_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
fig.suptitle('F1-Score per Class (Clear na Clear test, Edited na Edited test)', fontsize=18, fontweight='bold', y=0.95)

for row_idx, dataset in enumerate(datasets):
    _, _, _, _, X_te, y_te, num_classes = get_preprocessed_data(dataset)
    y_te_flat = y_te.flatten()

    for col_idx, rate in enumerate(rates):
        ax = axes[row_idx, col_idx]
        model_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_best.keras')

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            y_pred_probs = model.predict(X_te, batch_size=4096, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            report = classification_report(y_te_flat, y_pred, output_dict=True, zero_division=0)

            f1_scores = []
            classes = np.arange(num_classes)
            for c in classes:
                c_str = str(c)
                if c_str in report:
                    f1_scores.append(report[c_str]['f1-score'] * 100)
                else:
                    f1_scores.append(0.0)

            bars = ax.bar(classes, f1_scores, color='#00838f', width=0.6)

            ax.axhline(y=90, color='#1a237e', linestyle='--', linewidth=1.5, alpha=0.7)

            ax.set_ylim(0, 105)
            ax.set_xticks(classes)
            if dataset == 'edited':
                ax.set_xticklabels(['Kl. 0', 'Kl. 1', 'Kl. 2', 'Kl. 5', 'Kl. 6'])
            else:
                ax.set_xticklabels([f'Kl. {c}' for c in classes])
            ax.set_title(f'{dataset.upper()} | Dropout: {rate}', fontsize=14, fontweight='bold')
            ax.set_ylabel('F1 [%]' if col_idx == 0 else '', fontsize=12)
            ax.set_xlabel('Numer klasy' if row_idx == 1 else '', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.88)

save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'f1_score_per_class_bars.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Zapisano w: {save_path}")