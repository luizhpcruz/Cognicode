class SymbolicDNA:
    def __init__(self):
        self.genes = {
            'awareness': 0.5,
            'adaptability': 0.5,
            'stability': 0.5
        }

    def evolve(self, feedback):
        # Evolução diferenciada para cada gene
        self.genes['awareness'] += 0.01 * feedback
        self.genes['adaptability'] += 0.02 * (abs(feedback) - 0.5)
        self.genes['stability'] += -0.01 * abs(feedback)
        # Limita os valores entre 0 e 1
        for k in self.genes:
            self.genes[k] = max(0.0, min(1.0, self.genes[k]))

    @property
    def fingerprint(self):
        # Representação dos genes para seleção
        return ' '.join(f"{k}:{v:.3f}" for k, v in self.genes.items())
