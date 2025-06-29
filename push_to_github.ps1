# Script PowerShell para automatizar o push para o GitHub
# Execute este script na raiz do seu projeto

# Adiciona todos os arquivos
Write-Host "Adicionando arquivos ao Git..."
git add .

# Faz commit apenas se houver mudanças
if ((git status --porcelain).Length -gt 0) {
    $msg = Read-Host "Digite a mensagem do commit"
    git commit -m "$msg"
    Write-Host "Commit realizado."
} else {
    Write-Host "Nenhuma alteração para commitar."
}

# Garante que está no branch main
Write-Host "Mudando para o branch main..."
git branch -M main

# Adiciona o remoto se não existir
$remote = git remote
if (-not $remote) {
    $remoteUrl = Read-Host "Digite a URL do repositório remoto (ex: https://github.com/usuario/repositorio.git)"
    git remote add origin $remoteUrl
    Write-Host "Repositório remoto adicionado."
}

# Faz o push para o GitHub
Write-Host "Enviando para o GitHub..."
git push -u origin main
Write-Host "Push concluído!"
