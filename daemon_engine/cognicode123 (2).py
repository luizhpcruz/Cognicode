def ΛΣ(path):
    try:
        with open(path, 'r') as f:
            ϕ₀ = f.read()
        return ϕ₀
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
        return None

def Δϕ(ϕ₀):
    if ϕ₀ is None:
        return None
    # Processa os dados
    return ϕ₀.upper()

def ⟁Ψ(ϕ₀):
    if ϕ₀ is None:
        return None
    χ₀ = ϕ₀[::-1]
    return χ₀

def Ω📄(χ₀):
    if χ₀ is None:
        print("Nenhum resultado para gerar relatório.")
    else:
        print("Relatório:", χ₀)

if __name__ == "__main__":
    data = ΛΣ("indicadores.txt")
    processed = Δϕ(data)
    outcome = ⟁Ψ(processed)
    Ω📄(outcome)
