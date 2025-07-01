"""
SymbolicDNA: DNA simbólico e mecanismos de mutação para CogniCode.
"""
import json
import hashlib
import numpy as np

class SymbolicDNA:
    def __init__(self, genes, fitness_score=0.0, generation=0, is_user_dna=False):
        self.genes = genes
        self.fitness_score = fitness_score
        self.generation = generation
        self.fingerprint = self._calculate_fingerprint()
        self.is_user_dna = is_user_dna
    def _calculate_fingerprint(self):
        gene_str = json.dumps(self.genes, sort_keys=True)
        return hashlib.sha3_256(gene_str.encode()).hexdigest()[:16]
    def mutate(self):
        new_genes = self.genes.copy()
        for key in new_genes:
            if np.random.random() < 0.1:
                new_genes[key] += np.random.normal(0, 0.1)
        return SymbolicDNA(new_genes, fitness_score=0.0, generation=self.generation + 1, is_user_dna=False)
