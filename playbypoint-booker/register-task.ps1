param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$TargetDate,

    [string]$TaskName = 'PlayByPoint Pickleball Booker',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $projectDirectory 'config.json'

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Missing $configPath. Copy config.example.json to config.json and edit it first."
}

$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
$target = [DateTime]::ParseExact(
    $TargetDate,
    'yyyy-MM-dd',
    [Globalization.CultureInfo]::InvariantCulture
)
$releaseDate = $target.AddDays(-[int]$config.bookingWindowDays)
$runAt = $releaseDate.Date.AddHours(6).AddMinutes(58)

if ($runAt -le (Get-Date)) {
    throw "The scheduled start $runAt is in the past."
}

$mode = if ($DryRun) { '--dry-run' } else { '--live' }
$nodePath = (Get-Command node).Source
$arguments = "`"$projectDirectory\src\api-book.mjs`" --config `"$configPath`" --date $TargetDate $mode"
$action = New-ScheduledTaskAction `
    -Execute $nodePath `
    -Argument $arguments `
    -WorkingDirectory $projectDirectory
$trigger = New-ScheduledTaskTrigger -Once -At $runAt
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Starts early, prewarms PlayByPoint, and books $TargetDate at the 7:00 AM release." `
    -Force | Out-Null

Write-Host "Registered '$TaskName' for $runAt."
Write-Host "Target reservation date: $TargetDate"
Write-Host "Mode: $mode"
