# CogniCode: Framework de Simulação e IA Evolutiva

Este repositório contém o núcleo do CogniCode, um framework para simulação, evolução e análise de inteligência artificial simbólica.

## Descrição
O CogniCode orquestra simulações evolutivas, incentivos e análise de agentes simbólicos. O sistema permite a criação, evolução e avaliação de DNAs simbólicos, integrando mecanismos de incentivo e uma DSL (Domain Specific Language) para interação dinâmica.

## Como rodar

1. Instale as dependências:
   ```bash
   pip install numpy
   ```
2. Execute o script principal:
   ```bash
   python main_cognicode.py
   ```
3. Interaja via comandos DSL no terminal. Para sair, digite `EXIT`.

## Funcionalidades
- Evolução simbólica de agentes (DNAs)
- Sistema de incentivos e tokens
- Interpretação de comandos via DSL customizada
- Modularidade para expansão de simulações e análises

## Organização
- `main_cognicode.py`: script principal do framework
- `evolution/`: pipeline de evolução e manipulação de DNAs
- `incentives/`: sistema de incentivos e tokens
- `symbolic_ai/`: modelos de DNA simbólico
- `cli/`: interpretador de comandos DSL

## Como contribuir
1. Faça um fork do projeto
2. Crie uma branch para sua feature
3. Envie um pull request

## Licença
Este projeto está sob a licença MIT.

---
Projeto experimental aberto para colaboração, pesquisa e inovação em IA simbólica e evolução artificial.
