Start-Process 'C:\Users\czyhed\AppData\Local\Programs\Xmind\Xmind.exe'
Start-Sleep -Seconds 5
$p = Get-Process -Name *xmind* -ErrorAction SilentlyContinue
if ($p) {
    Write-Host "XMind started successfully - PID: $($p[0].Id)"
} else {
    Write-Host "XMind failed to start"
}
