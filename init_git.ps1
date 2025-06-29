# Script PowerShell para automatizar o setup do Git
# Execute este script na raiz do seu projeto

# Inicializa o repositório Git
if (-not (Test-Path .git)) {
    git init
    Write-Host "Repositório Git inicializado."
    # Altera o branch padrão para main
    git branch -m main
} else {
    Write-Host "Repositório Git já existe."
}

# Adiciona todos os arquivos
Write-Host "Adicionando arquivos ao Git..."
git add .

# Faz o commit inicial
Write-Host "Fazendo commit inicial..."
git commit -m "Commit inicial do projeto automatizado"

# Pergunta se deseja adicionar um repositório remoto
$defaultRemote = Read-Host "Deseja adicionar um repositório remoto? (s/n)"
if ($defaultRemote -eq 's') {
    $remoteUrl = Read-Host "Digite a URL do repositório remoto"
    git remote add origin $remoteUrl
    Write-Host "Repositório remoto adicionado."
    Write-Host "Para enviar os arquivos, execute: git push -u origin main"
} else {
    Write-Host "Setup Git concluído sem repositório remoto."
}
