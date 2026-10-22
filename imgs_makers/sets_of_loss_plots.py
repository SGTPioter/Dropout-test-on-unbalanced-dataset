import os
import pandas as pd
import matplotlib.pyplot as plt

# Dynamiczne ustalenie głównego folderu projektu (P1) - cofamy się o jeden poziom od lokalizacji skryptu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharey=True)
fig.suptitle('Przebieg uczenia (Funkcja straty) w zależności od wariantu zbioru i Dropoutu',
             fontsize=18, fontweight='bold', y=0.97)

for col_idx, dataset in enumerate(datasets):
    for row_idx, rate in enumerate(rates):
        ax = axes[row_idx, col_idx]

        # Odczyt logów z głównego folderu models/
        csv_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_history.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            epochs = df['epoch'] + 1 if 'epoch' in df.columns else df.index + 1

            ax.plot(epochs, df['loss'], label='Trening', color='#1f77b4', linewidth=1.8)
            ax.plot(epochs, df['val_loss'], label='Walidacja', color='#d62728', linewidth=1.8)

            min_val_idx = df['val_loss'].idxmin()
            min_val_epoch = epochs.iloc[min_val_idx] if hasattr(epochs, 'iloc') else epochs[min_val_idx]
            min_val_loss = df['val_loss'].iloc[min_val_idx]

            ax.axvline(x=min_val_epoch, color='grey', linestyle=':', alpha=0.8,
                       label=f'Min. błędu (epoka {min_val_epoch})')
            ax.scatter(min_val_epoch, min_val_loss, color='#d62728', zorder=5)

        ax.set_title(f'Zbiór: {dataset.upper()}  |  Dropout: {rate}', fontsize=13)
        ax.set_xlabel('Epoka')
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.legend(loc='upper right')

        if col_idx == 0:
            ax.set_ylabel('Sparse Categorical Crossentropy')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Zapis do głównego folderu imgs/
save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'sets_of_loss_plots.png')

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Wygenerowano wykres i zapisano w: {save_path}")