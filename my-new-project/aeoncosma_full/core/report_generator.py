# Estrutura principal do novo módulo: aeoncosma/modules/report_generator.py

import os
import numpy as np
from datetime import datetime

LATEX_TEMPLATE = r"""
\\documentclass[a4paper,12pt]{{article}}
\\usepackage{{amsmath,graphicx,float}}
\\title{{Relat\'orio AEONCOSMA}}
\\date{{{date}}}
\\begin{document}
\\maketitle

\\section*{{Resumo Estat\'istico}}

\\begin{{itemize}}
  \item $\chi^2$: {chi2:.2f}
  \item AIC: {aic:.2f}
  \item BIC: {bic:.2f}
\\end{{itemize}}

\\section*{{Par\^ametros Estimados}}
\\begin{{tabular}}{{|c|c|}}
\\hline
Par\^ametro & Valor \\
\\hline
{param_lines}\\end{{tabular}}

\\section*{{Gr\'afico do Ajuste}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\textwidth]{{ajuste.png}}
\\caption{Curva ajustada aos dados}
\\end{{figure}}

\\end{{document}}
"""

def gerar_relatorio_latex(resultados, caminho_saida="aeoncosma/output"):
    os.makedirs(caminho_saida, exist_ok=True)

    param_lines = ""
    for nome, valor in resultados['parametros'].items():
        param_lines += f"{nome} & {valor:.4f} \\\\ 
\\hline\n"

    conteudo = LATEX_TEMPLATE.format(
        date=datetime.now().strftime("%d/%m/%Y"),
        chi2=resultados["chi2"],
        aic=resultados["aic"],
        bic=resultados["bic"],
        param_lines=param_lines
    )

    with open(os.path.join(caminho_saida, "relatorio.tex"), "w") as f:
        f.write(conteudo)
    print("✅ LaTeX gerado em output/relatorio.tex")

# Exemplo de uso (em outro script ou notebook):
# resultados = {
#     "chi2": 101.3,
#     "aic": 111.3,
#     "bic": 116.2,
#     "parametros": {
#         "H0": 68.34,
#         "Omega_m": 0.302,
#         "Omega_ond": 0.15,
#         "n": -1.7,
#     }
# }
# gerar_relatorio_latex(resultados)

# Observação: você ainda precisa compilar com pdflatex para gerar o PDF.
