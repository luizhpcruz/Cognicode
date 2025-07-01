def load_data(path):
    try:
        with open(path, 'r') as f:
            dataset = f.read()
        return dataset
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
        return None

def process_data(dataset):
    if dataset is None:
        return None
    # Processa os dados
    return dataset.upper()

def run_simulation(dataset):
    if dataset is None:
        return None
    result = dataset[::-1]
    return result

def generate_report(result):
    if result is None:
        print("Nenhum resultado para gerar relatório.")
    else:
        print("Relatório:", result)

if __name__ == "__main__":
    data = load_data("indicadores.txt")
    processed = process_data(data)
    outcome = run_simulation(processed)
    generate_report(outcome)
