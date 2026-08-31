# Kill all XMind processes
Stop-Process -Name Xmind -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Verify
$remaining = Get-Process -Name *xmind* -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "STILL RUNNING:"
    $remaining | Select-Object Id, ProcessName | Format-Table -AutoSize
} else {
    Write-Host "ALL XMind processes killed successfully"
}

# Check the problematic account.json file
$stateFile = Join-Path $env:APPDATA 'Xmind\Electron v3\vana\state\account.json'
if (Test-Path $stateFile) {
    Write-Host ""
    Write-Host "=== account.json info ==="
    Get-ItemProperty $stateFile | Select-Object FullName, Length, Attributes, LastWriteTime | Format-List
    
    # Check if file is locked
    try {
        $stream = [System.IO.File]::Open($stateFile, 'Open', 'ReadWrite', 'None')
        $stream.Close()
        Write-Host "File is NOT locked - accessible"
    } catch {
        Write-Host "File IS LOCKED: $($_.Exception.Message)"
    }
    
    # Backup and remove the problematic file
    $backupFile = "$stateFile.bak"
    Copy-Item -Path $stateFile -Destination $backupFile -Force
    Remove-Item -Path $stateFile -Force
    Write-Host "Backed up to account.json.bak and removed original"
} else {
    Write-Host "account.json does not exist"
}
