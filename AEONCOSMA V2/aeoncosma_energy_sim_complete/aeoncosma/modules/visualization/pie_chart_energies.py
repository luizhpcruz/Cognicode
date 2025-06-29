
import matplotlib.pyplot as plt

def plot_pie(Omega_r, Omega_m, Omega_Lambda):
    labels = ['Radiação', 'Matéria', 'Energia Escura']
    sizes = [Omega_r, Omega_m, Omega_Lambda]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title("Composição energética atual")
    plt.savefig("energy_pie.png")
    plt.close()
