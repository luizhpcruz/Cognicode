import argparse
import sys

# CogniCode CLI: Geração de código a partir de prompts em linguagem natural

def main():
    parser = argparse.ArgumentParser(description='CogniCode - Geração de código via prompt CLI')
    parser.add_argument('prompt', type=str, help='Prompt em linguagem natural para gerar código')
    parser.add_argument('--lang', type=str, default='python', help='Linguagem de programação (padrão: python)')
    parser.add_argument('--output', type=str, help='Arquivo para salvar o código gerado (opcional)')
    args = parser.parse_args()

    # Exemplo de resposta simulada (substitua por integração com modelo de IA futuramente)
    if 'energia potencial' in args.prompt.lower() and args.lang == 'python':
        code = '''def energia_potencial(massa, gravidade, altura):
    """Calcula a energia potencial gravitacional."""
    return massa * gravidade * altura
'''
    elif 'plotar a evolução do fitness' in args.prompt.lower() and args.lang == 'python':
        code = '''import matplotlib.pyplot as plt\n\ndef plot_fitness(fitness):\n    plt.plot(fitness)\n    plt.xlabel(\'Geração\')\n    plt.ylabel(\'Fitness\')\n    plt.title(\'Evolução do Fitness\')\n    plt.show()\n'''
    else:
        code = f'# [Simulação] Código em {args.lang} para: {args.prompt}\npass\n'

    print(code)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(code)
        print(f'Código salvo em {args.output}')

if __name__ == '__main__':
    main()
