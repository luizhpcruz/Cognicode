# Estrutura do Documento LaTeX

## 1. Preâmbulo
   - Classe do documento: article (ou similar, como revtex para física)
   - Pacotes essenciais: geometry, graphicx, amsmath, amssymb, hyperref, caption, booktabs (para tabelas)
   - Configurações de página e idioma (português)

## 2. Seção de Título
   - Título: Análise Robusta do Modelo ΛCDM com Dados H(z) de Cosmic Chronometers
   - Autor: (A ser definido, pode ser o usuário ou uma colaboração)
   - Data: \today

## 3. Resumo (Abstract)
   - Breve descrição do contexto (ΛCDM, Tensão de Hubble).
   - Objetivo: Reanalisar dados H(z) de CC com tratamento rigoroso de erros.
   - Metodologia: 15 pontos de dados (Moresco et al.), matriz de covariância completa (estatística + sistemática), ajuste MCMC para ΛCDM plano.
   - Resultados chave: H₀ ≈ 63 ± 10 km/s/Mpc, Ω<0xE2><0x82><0x98> ≈ 0.35 ± 0.11, consistência com Planck/BAO dentro das incertezas.

## 4. Seção 1: Introdução
   - Apresentar o modelo ΛCDM como padrão.
   - Mencionar a Tensão de Hubble como motivação.
   - Introduzir os Cosmic Chronometers (CC) como sonda independente.
   - Declarar o objetivo do trabalho: realizar um ajuste robusto do ΛCDM aos dados CC mais recentes, utilizando a matriz de covariância completa.

## 5. Seção 2: Dados e Metodologia
   - **Subseção 2.1: Dados H(z):**
     - Descrever a fonte dos dados (Moresco et al., 15 pontos de `HzTable_MM_BC03.dat`).
     - Opcional: Incluir uma tabela com os valores (z, H(z), σ_stat).
   - **Subseção 2.2: Matriz de Covariância:**
     - Explicar a necessidade da matriz completa (Cov = Cov_stat + Cov_syst).
     - Detalhar as componentes sistemáticas consideradas (IMF, stlib, SPS, met - de `hz_data_moresco2020.dat` e `HzTable_MM_BC03.dat`).
     - Mencionar o arquivo gerado (`cov_matrix_total_hz.dat`).
   - **Subseção 2.3: Modelo Cosmológico:**
     - Apresentar a equação H(z) para o modelo ΛCDM plano.
   - **Subseção 2.4: Análise Estatística:**
     - Descrever o método MCMC (pacote `emcee`).
     - Definir a função de log-verossimilhança (χ² com a inversa da covariância).
     - Especificar os priors (planos e largos para H₀ e Ω<0xE2><0x82><0x98>).
     - Mencionar os critérios AIC e BIC.

## 6. Seção 3: Resultados
   - Apresentar os valores medianos e intervalos de 68% para H₀ e Ω<0xE2><0x82><0x98> (de `mcmc_fit_summary_hz_fullcov.txt`).
   - Incluir o gráfico de contorno (corner plot) (`mcmc_corner_plot_hz_fullcov_corrected.png`) usando `figure` e `includegraphics`.
   - Reportar χ²_min, χ²/dof, AIC, BIC.
   - Opcional: Tabela resumindo os resultados estatísticos.

## 7. Seção 4: Discussão
   - Comparar H₀ e Ω<0xE2><0x82><0x98> com Planck 2018, Pantheon+ SNe Ia, BAO.
   - Discutir a consistência/tensão (H₀ baixo mas com grande erro, Ω<0xE2><0x82><0x98> consistente).
   - Contrastar com a análise preliminar (N=5, sem covariância) para ilustrar o impacto da metodologia.
   - Comentar o valor do χ²/dof.

## 8. Seção 5: Conclusão
   - Recapitular os principais resultados da análise robusta.
   - Enfatizar a importância do tratamento correto dos erros e covariâncias.
   - Concluir sobre a consistência dos dados CC (com grandes erros) com o ΛCDM padrão.

## 9. Referências (Opcional)
   - Citar os trabalhos de Moresco et al. relevantes.
   - Citar Planck Collaboration, Pantheon+, etc.

## 10. Apêndice (Opcional)
   - Tabela completa dos dados H(z).
   - Detalhes adicionais sobre a matriz de covariância.

