from universe import Universe

def run_simulation(num_universes=3, total_simulation_time=20.0, dt_multiplier=50):
    universes = []

    for i in range(num_universes):
        parent_dna = universes[0].cosmic_dna.fingerprint if i > 0 else None
        universe = Universe(
            universe_id=i,
            initial_conditions={'matter_density': 0.3, 'radiation_density': 0.001},
            vibrational_frequency=1.0 + i * 0.1,
            parent_dna=parent_dna
        )
        universes.append(universe)

    dt = 1.0 / dt_multiplier
    steps = int(total_simulation_time * dt_multiplier)

    print("Iniciando simulação dos universos...")

    for step in range(steps):
        for universe in universes:
            universe.evolve_universe(time_step=dt, all_universes=universes)

        if step % (dt_multiplier // 2) == 0:
            print(f"Tempo simulado: {step * dt:.2f}")
            for u in universes:
                print(f" Universo {u.universe_id}: H = {u.hubble_constant:.4e}, phi = {u.cosmic_resonance.phi:.4f}, Energia_phi = {u.cosmic_resonance.get_energy_density_phi():.4e}")

    print("Simulação finalizada.")

if __name__ == "__main__":
    run_simulation()
