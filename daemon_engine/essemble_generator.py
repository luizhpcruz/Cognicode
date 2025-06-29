from simulation_universe import Universe
import numpy as np

def generate_universe_ensemble(num_universes: int, initial_conditions: dict, vibrational_frequencies: list, parent_dna: str = None) -> list:
    """Gera um conjunto de universos com diferentes frequências e DNAs."""
    universes = []
    for i in range(num_universes):
        # O DNA pai é herdado apenas do primeiro universo da população, se houver
        dna_to_inherit = parent_dna if parent_dna and i > 0 else None
        freq = vibrational_frequencies[i % len(vibrational_frequencies)]
        universe = Universe(
            universe_id=i,
            initial_conditions=initial_conditions,
            vibrational_frequency=freq,
            parent_dna=dna_to_inherit
        )
        universes.append(universe)
    return universes