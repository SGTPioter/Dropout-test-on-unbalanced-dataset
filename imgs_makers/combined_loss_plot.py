import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets = ['clear', 'edited']
rates = [0.0, 0.2, 0.5]

colors = {
    'clear': {
        0.0: '#64b5f6',  # Jasnoniebieski
        0.2: '#1976d2',  # Niebieski
        0.5: '#0d47a1'  # Ciemnoniebieski
    },
    'edited': {
        0.0: '#e57373',  # Jasnoczerwony
        0.2: '#d32f2f',  # Czerwony
        0.5: '#b71c1c'  # Ciemnoczerwony
    }
}

plt.figure(figsize=(12, 8))
plt.title('Zbiorcze porównanie funkcji straty (Walidacja) dla wszystkich 6 modeli',
          fontsize=16, fontweight='bold', pad=15)

# Przejście przez wszystkie modele i rysowanie krzywych
for dataset in datasets:
    for rate in rates:
        csv_path = os.path.join(BASE_DIR, 'models', dataset, f'model_dropout_{rate}_history.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            epochs = df['epoch'] + 1 if 'epoch' in df.columns else df.index + 1

            label_name = f"{dataset.capitalize()} - Dropout {rate}"
            line_color = colors[dataset][rate]

            plt.plot(epochs, df['val_loss'], label=label_name, color=line_color, linewidth=2)

plt.xlabel('Epoka', fontsize=12)
plt.ylabel('Sparse Categorical Crossentropy (Validation Loss)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Warianty modeli:', title_fontsize=12, fontsize=10, loc='upper right')

save_dir = os.path.join(BASE_DIR, 'imgs')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'combined_loss_plot.png')

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Wygenerowano wspólny wykres i zapisano w: {save_path}")