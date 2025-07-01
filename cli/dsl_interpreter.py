"""
Interpretador DSL para o CogniCode (CLI).
"""
def run_dsl_script(script, evolution_pipeline, sistema_avaliacao):
    for line in script.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        command = (
            tokens[0]
            .upper()
            .replace("Ç", "C")
            .replace("Ã", "A")
            .replace("Õ", "O")
            .replace("É", "E")
            .replace("Ê", "E")
            .replace("Á", "A")
            .replace("Í", "I")
            .replace("Ó", "O")
            .replace("Ú", "U")
            .replace("_", "")
        )
        args = tokens[1:]
        print(f"Executando Comando DSL: {command}")
        try:
            if command == "GETPONTUACAO":
                if args:
                    agente_id = args[0]
                    pontos = sistema_avaliacao.get_pontuacao(agente_id)
                    print(f"Pontuação para {agente_id[:8]}...: {pontos:.4f} pontos.")
                else:
                    print("Uso: GET_PONTUACAO <agente_id>")
            elif command == "REGISTRARAGENTE":
                if args:
                    agente_id = args[0]
                    sistema_avaliacao.registrar_agente(agente_id)
                else:
                    print("Uso: REGISTRAR_AGENTE <agente_id>")
            elif command == "SIMULAREVOLUCAO":
                # Simula uma rodada evolutiva (exemplo: apenas incrementa a época)
                evolution_pipeline.current_epoch += 1
                print(f"Simulação evolutiva rodada. Época atual: {evolution_pipeline.current_epoch}")
            else:
                print(f"Comando desconhecido: {command}")
        except Exception as e:
            print(f"Erro ao executar comando '{command}': {e}")
