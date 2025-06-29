"""Geração automática de relatório em LaTeX"""
def gerar_latex(params, output="relatorio.tex"):
    """
    Gera um arquivo LaTeX com os parâmetros e resultados da simulação.
    
    Args:
        params: Dicionário de parâmetros a serem incluídos no relatório.
        output: Nome do arquivo de saída.
    """
    with open(output, 'w') as f:
        f.write("\documentclass{article}\n\\begin{document}\n\\section*{Resultados da Simulação}\n")
        for k, v in params.items():
            f.write(f"\\textbf{{{k}}}: {v}\\\\\n")
        f.write("\end{document}")