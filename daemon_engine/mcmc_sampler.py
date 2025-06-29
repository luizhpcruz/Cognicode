"""Executa o MCMC com emcee"""
import emcee
import numpy as np

def run_sampler(log_prob_fn, initial_pos, nwalkers, steps):
    """
    Roda o sampler MCMC para estimar os parâmetros do modelo.
    
    Args:
        log_prob_fn: Função de probabilidade logarítmica (verossimilhança + prior).
        initial_pos: Posições iniciais dos 'walkers'.
        nwalkers: Número de 'walkers'.
        steps: Número de passos da cadeia.
    
    Returns:
        O objeto sampler do emcee com os resultados.
    """
    sampler = emcee.EnsembleSampler(nwalkers, len(initial_pos), log_prob_fn)
    sampler.run_mcmc(initial_pos, steps, progress=True)
    return sampler