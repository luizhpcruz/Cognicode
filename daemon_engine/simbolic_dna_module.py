import numpy as np
import random
import json
import hashlib

class SymbolicDNA:
    """
    Representa o 'DNA simbólico' de um universo ou sistema, com genes e um fingerprint único.
    """
    def __init__(self, genes=None, fitness_score=0.0, generation=0):
        if genes:
            self.genes = genes
        else:
            # Genes representam parâmetros simbólicos que podem ser herdados e mutados
            self.genes = {
                'bio_diversity_focus': random.uniform(0.1, 0.9),
                'tech_innovation_focus': random.uniform(0.1, 0.9),
                'social_cohesion_focus': random.uniform(0.1, 0.9),
                'adaptability_gene': random.uniform(0.1, 0.9),
                'collective_awareness_drive': random.uniform(0.1, 0.9)
            }
        self.fitness_score = fitness_score
        self.generation = generation
        self.fingerprint = self._calculate_fingerprint()

    def _calculate_fingerprint(self) -> str:
        """
        Gera um hash único para o DNA, atuando como um 'fingerprint'.
        """
        gene_str = json.dumps(self.genes, sort_keys=True)
        return hashlib.sha3_256(gene_str.encode()).hexdigest()[:16]

    def mutate(self) -> 'SymbolicDNA':
        """
        Cria uma nova versão do DNA com mutações probabilísticas nos genes.
        """
        new_genes = self.genes.copy()
        for key in new_genes:
            if np.random.random() < 0.1: # 10% de chance de mutação por gene
                new_genes[key] += np.random.normal(0, 0.1)
                new_genes[key] = max(0.0, min(1.0, new_genes[key])) # Limita o valor entre 0 e 1
        return SymbolicDNA(new_genes, generation=self.generation + 1)

    def evolve(self, success_metric: float):
        """
        Evolui o DNA com base em uma métrica de sucesso (ex: entropia).
        """
        # Exemplo simples: muta se a performance for baixa
        if success_metric < 0.5:
            mutated = self.mutate()
            self.genes = mutated.genes
            self.fingerprint = mutated.fingerprint
            self.generation += 1
            self.fitness_score = 0.0