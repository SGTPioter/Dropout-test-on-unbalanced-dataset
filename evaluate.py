import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(csv_path, rate):
    """
    Rysuje wykresy Loss i Accuracy dla pojedynczego wariantu modelu.
    """
    df = pd.read_csv(csv_path)

    # Wykres z dwoma panelami: obok siebie Loss i Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Funkcja straty (Loss)
    ax1.plot(df['epoch'], df['loss'], label='Trening (Loss)', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Walidacja (Val Loss)', linewidth=2)

    # Zaznaczenie momentu, w którym walidacja była najlepsza
    min_val_loss_epoch = df['val_loss'].idxmin()
    ax1.axvline(min_val_loss_epoch, color='gray', linestyle=':',
                label=f'Min Val Loss (epoka {min_val_loss_epoch})')

    ax1.set_title(f'Krzywe Uczenia (Loss) - Dropout: {rate}')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Sparse Categorical Crossentropy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Dokładność (Accuracy)
    ax2.plot(df['epoch'], df['accuracy'], label='Trening (Acc)', linewidth=2)
    ax2.plot(df['epoch'], df['val_accuracy'], label='Walidacja (Val Acc)', linewidth=2)
    ax2.set_title(f'Dokładność (Accuracy) - Dropout: {rate}')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/learning_curves_dropout_{rate}.png', dpi=300)
    plt.close()


def plot_validation_comparison(rates):
    """
    Rysuje zbiorczy wykres funkcji straty na zbiorze walidacyjnym
    dla wszystkich testowanych konfiguracji.
    """
    plt.figure(figsize=(10, 6))

    for rate in rates:
        csv_path = f"models/model_dropout_{rate}_history.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['val_loss'], label=f'Dropout {rate}', linewidth=2)

    plt.title('Porównanie Val Loss w zależności od współczynnika Dropout')
    plt.xlabel('Epoka')
    plt.ylabel('Val Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/comparison_val_loss.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    dropout_rates = [0.0, 0.2, 0.5]

    print("Generowanie wykresów krzywych uczenia...")
    for rate in dropout_rates:
        csv_path = f"models/model_dropout_{rate}_history.csv"
        if os.path.exists(csv_path):
            plot_learning_curves(csv_path, rate)
        else:
            print(f"Brak pliku {csv_path}. Upewnij się, że trening przebiegł pomyślnie.")

    print("Generowanie wykresu porównawczego...")
    plot_validation_comparison(dropout_rates)

    print("Wykresy zostały pomyślnie zapisane w folderze 'plots/'.")