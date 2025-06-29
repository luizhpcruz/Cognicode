"""Plot de H(z) com bandas de incerteza"""
import matplotlib.pyplot as plt

def plot_hubble(z, hz_values, label='Modelo', ax=None):
    """
    Plota a curva de H(z) com um rótulo.
    
    Args:
        z: Array de redshifts.
        hz_values: Array de valores de H(z).
        label: Rótulo para a curva.
        ax: Objeto Axes para plotagem em subplots.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(z, hz_values, label=label)
    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("H(z)")
    ax.legend()
    ax.grid(True)
    
    if ax is None:
        plt.show()