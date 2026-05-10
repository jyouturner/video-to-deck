# yt2md installer for Windows — turn long YouTube videos into readable
# digests + slides. Run with:
#
#   irm https://raw.githubusercontent.com/jyouturner/youtube-to-markdown/main/install.ps1 | iex
#
# This script does four things:
#   1. Verifies you're on Windows 10/11 with PowerShell 5+.
#   2. Verifies winget is installed (built into Windows 10 1809+ via the
#      App Installer Store package — prints a link if missing).
#   3. winget installs ffmpeg, Node.js, and uv (no-ops if already present).
#   4. uv tool installs yt2md from this repo's main branch.
#
# Re-running this script upgrades yt2md to the latest commit on main.
# Source you're about to run:
#   https://github.com/jyouturner/youtube-to-markdown/blob/main/install.ps1
#
# yt2md's Windows support is currently untested; please open an issue if
# something breaks. Curl-bash equivalent for macOS users lives at install.sh.

#Requires -Version 5.1

$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/jyouturner/youtube-to-markdown'
$InstallSource = "git+$RepoUrl"
$PackageName = 'youtube-to-markdown'

# Color-aware logging helpers. Out-Host with -ForegroundColor lets us add
# subtle visual structure without depending on Write-Host (which is
# considered an anti-pattern for piped scripts but is fine here since
# nothing downstream consumes our stdout).
function Write-Heading($text)  { Write-Host ""; Write-Host $text -ForegroundColor White }
function Write-Ok($text)        { Write-Host "  $([char]0x2713) $text" -ForegroundColor Green }
function Write-Info($text)      { Write-Host "  $text" -ForegroundColor DarkGray }
function Write-Fail($text, $hint) {
    Write-Host ""
    Write-Host "  $([char]0x2717) $text" -ForegroundColor Red
    if ($hint) { Write-Host "  -> $hint" -ForegroundColor DarkGray }
    Write-Host ""
    exit 1
}

Write-Heading "yt2md installer (Windows)"
Write-Info "Source: $RepoUrl"

# 1. Platform: Windows + PowerShell 5+ (which is built into Windows 10/11).
Write-Heading "1/4  Platform"
if (-not $IsWindows -and $PSVersionTable.PSEdition -ne 'Desktop') {
    # PowerShell Core on non-Windows. Refuse with a hint to use the bash
    # installer instead — this script's package-manager assumptions are
    # winget-specific.
    Write-Fail "This installer only supports Windows." `
               "macOS / Linux: use install.sh from the same repo."
}
$os = (Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction SilentlyContinue).Caption
if (-not $os) { $os = "Windows" }
Write-Ok "$os (PowerShell $($PSVersionTable.PSVersion))"

# 2. winget. Built into modern Windows but not into older builds; the user
# can install it via the Microsoft Store ("App Installer") if missing.
Write-Heading "2/4  winget"
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Fail "winget not found." `
               "Install 'App Installer' from the Microsoft Store, then re-run this script."
}
$wingetVersion = (winget --version) 2>$null
Write-Ok "found ($wingetVersion)"

# 3. winget install ffmpeg + Node.js + uv. winget's exit codes are nuanced:
# 0 = installed, an installer-specific code = "already installed at same
# version" — both should be treated as success here. We use --silent to
# suppress GUI prompts and --accept-* to keep things hands-off.
Write-Heading "3/4  System dependencies"
$packages = @(
    @{ Id = 'Gyan.FFmpeg';        Name = 'ffmpeg' },
    @{ Id = 'OpenJS.NodeJS';      Name = 'Node.js' },
    @{ Id = 'astral-sh.uv';       Name = 'uv'      }
)
foreach ($pkg in $packages) {
    Write-Info "winget installing $($pkg.Name)..."
    $null = winget install --exact --id $pkg.Id `
        --accept-package-agreements --accept-source-agreements `
        --silent 2>$null
    # winget exits 0 on success and 0x8A150061 (-1978335135) on
    # "already installed at the same version". Both fine. Anything else
    # we surface — but most failures here mean missing networking or a
    # blocked source, not a real error in our script.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335135) {
        Write-Info "  (winget returned $LASTEXITCODE — assuming already installed)"
    }
    Write-Ok "$($pkg.Name) ready"
}

# winget installs put binaries into directories that are usually added to
# PATH automatically, but the *current* PowerShell session won't see them
# until restart. Refresh PATH from the registry so the rest of this script
# can call uv directly.
$env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path', 'User')

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Fail "uv installed but not on PATH for this session." `
               "Open a new PowerShell window and re-run the curl-bash command."
}

# 4. yt2md itself, via uv tool install. --reinstall makes the script
# upgrade-by-default on subsequent runs.
Write-Heading "4/4  yt2md"
$uvList = (uv tool list 2>$null) -join "`n"
if ($uvList -match "(?m)^$([regex]::Escape($PackageName))\s") {
    Write-Info "yt2md already installed - upgrading to latest main..."
    uv tool install --reinstall $InstallSource
} else {
    Write-Info "uv tool installing from $InstallSource..."
    uv tool install $InstallSource
}
if ($LASTEXITCODE -ne 0) {
    Write-Fail "uv tool install failed (exit $LASTEXITCODE)." `
               "Try running 'uv tool install $InstallSource' directly to see the full error."
}
Write-Ok "yt2md installed"

# Done.
Write-Heading "Done"
Write-Host ""
Write-Host "  Run " -NoNewline
Write-Host "yt2md serve" -ForegroundColor Cyan -NoNewline
Write-Host " to start the local reader."
Write-Host ""
Write-Host "  First run opens a setup page in your browser asking for either:"
Write-Host "    - an Anthropic API key  (https://console.anthropic.com/settings/keys), or"
Write-Host "    - a Claude.ai Pro/Max login via the bundled Claude Code sandbox."
Write-Host ""
Write-Host "  Re-run this irm | iex command to upgrade yt2md to the latest version." -ForegroundColor DarkGray
Write-Host ""
