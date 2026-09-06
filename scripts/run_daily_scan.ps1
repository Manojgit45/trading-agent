$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) {
    throw "Python virtual environment not found at $pythonExe. Create it first: python -m venv .venv"
}

$args = @(
    '.\daily_scanner.py',
    '--universe', '.\market_universe.txt',
    '--top', '10',
    '--output', '.\top_10_stocks.json',
    '--csv-output', '.\top_10_stocks.csv'
)

if ($env:TELEGRAM_BOT_TOKEN -and $env:TELEGRAM_CHAT_ID) {
    $args += @('--telegram-token', $env:TELEGRAM_BOT_TOKEN, '--telegram-chat-id', $env:TELEGRAM_CHAT_ID)
}

if ($env:SMTP_SERVER -and $env:SMTP_USER -and $env:SMTP_PASSWORD -and $env:EMAIL_FROM -and $env:EMAIL_TO) {
    $args += @(
        '--smtp-server', $env:SMTP_SERVER,
        '--smtp-port', ($env:SMTP_PORT ?? '587'),
        '--smtp-user', $env:SMTP_USER,
        '--smtp-password', $env:SMTP_PASSWORD,
        '--email-from', $env:EMAIL_FROM,
        '--email-to'
    )
    $args += $env:EMAIL_TO.Split(',')
}

& $pythonExe $args
