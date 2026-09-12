param(
    [switch]$Remove,
    [ValidatePattern('^(?:[01][0-9]|2[0-3]):[0-5][0-9]$')]
    [string]$At = '18:00'
)
$ErrorActionPreference = 'Stop'
$task6Root = Split-Path -Parent $PSScriptRoot
$task6Python = Join-Path $task6Root '.venv\Scripts\python.exe'
$task6Script = Join-Path $PSScriptRoot 'update_index.py'
$task6Name = 'Sprint7-Task6-DailyIndex'
$task6Arguments = '"' + $task6Script + '"'
if (-not (Test-Path -LiteralPath $task6Python)) { throw 'Project Python was not found.' }
$task6Existing = Get-ScheduledTask -TaskName $task6Name -ErrorAction SilentlyContinue
if ($task6Existing) {
    if ($task6Existing.Actions.Count -ne 1 -or
        $task6Existing.Actions[0].Execute -ne $task6Python -or
        $task6Existing.Actions[0].Arguments -ne $task6Arguments) {
        throw 'A different task already uses this name; nothing changed.'
    }
}
if ($Remove) {
    if ($task6Existing) { Unregister-ScheduledTask -TaskName $task6Name -Confirm:$false }
    Write-Output 'Task 6 schedule removed.'
    exit
}
$task6Action = New-ScheduledTaskAction -Execute $task6Python -Argument $task6Arguments -WorkingDirectory $task6Root
$task6Trigger = New-ScheduledTaskTrigger -Daily -At $At
$task6Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task6Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$task6Principal = New-ScheduledTaskPrincipal -UserId $task6Identity -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $task6Name -Action $task6Action -Trigger $task6Trigger `
    -Settings $task6Settings -Principal $task6Principal -Force | Out-Null
Write-Output "Registered daily at $At local time, for the current logged-in user."
Write-Output 'Missed run: StartWhenAvailable; failures: two retries, five minutes apart.'
