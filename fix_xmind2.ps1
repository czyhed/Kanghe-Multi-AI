$stateDir = Join-Path $env:APPDATA 'Xmind\Electron v3\vana\state'

# Check for readonly files in state directory
Write-Host "=== ReadOnly files in state directory ==="
$readonlyFiles = Get-ChildItem $stateDir -File -ErrorAction SilentlyContinue | Where-Object { $_.Attributes -match 'ReadOnly' }
if ($readonlyFiles) {
    $readonlyFiles | ForEach-Object {
        Write-Host "Fixing: $($_.Name)"
        $_.Attributes = 'Normal'
    }
    Write-Host "Fixed $($readonlyFiles.Count) readonly files"
} else {
    Write-Host "No readonly files found"
}

# Check for lock files
Write-Host ""
Write-Host "=== Lock files in Xmind data ==="
$lockFiles = Get-ChildItem (Join-Path $env:APPDATA 'Xmind') -Recurse -Filter '*.lock' -ErrorAction SilentlyContinue
if ($lockFiles) {
    $lockFiles | ForEach-Object {
        Write-Host "Removing lock: $($_.FullName)"
        Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "No lock files found"
}

# Check for SingletonLock (Electron single instance lock)
$electronDir = Join-Path $env:APPDATA 'Xmind\Electron v3'
$singletonLock = Join-Path $electronDir 'SingletonLock'
if (Test-Path $singletonLock) {
    Remove-Item $singletonLock -Force -ErrorAction SilentlyContinue
    Write-Host "Removed SingletonLock"
}

$singletonCookie = Join-Path $electronDir 'SingletonCookie'
if (Test-Path $singletonCookie) {
    Remove-Item $singletonCookie -Force -ErrorAction SilentlyContinue
    Write-Host "Removed SingletonCookie"
}

$singletonSocket = Join-Path $electronDir 'SingletonSocket'
if (Test-Path $singletonSocket) {
    Remove-Item $singletonSocket -Force -ErrorAction SilentlyContinue
    Write-Host "Removed SingletonSocket"
}

Write-Host ""
Write-Host "=== Cleanup complete. Ready to launch XMind. ==="
