Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$templatePath = Join-Path $projectDirectory 'config.example.json'
$configPath = Join-Path $projectDirectory 'config.json'
$nodePath = (Get-Command node -ErrorAction Stop).Source
$loginScript = Join-Path $projectDirectory 'src\login.mjs'
$apiScript = Join-Path $projectDirectory 'src\api-book.mjs'
$logsDirectory = Join-Path $projectDirectory 'logs'
New-Item -ItemType Directory -Force -Path $logsDirectory | Out-Null

$script:LoginProcess = $null
$script:ArmedProcess = $null
$script:ArmedOutputOffset = 0
$script:ArmedOutputPending = ''
$script:SuccessDetails = $null
$script:HealthCheckRequestedAt = $null
$script:PreflightPassed = $false
$script:ArmedOutput = Join-Path $logsDirectory 'gui-armed.out.log'
$script:ArmedError = Join-Path $logsDirectory 'gui-armed.err.log'
$script:HealthCheckRequest = Join-Path $logsDirectory 'gui-healthcheck.request.json'

function Load-ReservationConfig {
    $source = if (Test-Path -LiteralPath $configPath) {
        $configPath
    } else {
        $templatePath
    }
    return Get-Content -Raw -LiteralPath $source | ConvertFrom-Json
}

function Add-Label {
    param(
        [System.Windows.Forms.Control]$Parent,
        [string]$Text,
        [int]$X,
        [int]$Y,
        [int]$Width = 140
    )
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.Location = New-Object System.Drawing.Point($X, $Y)
    $label.Size = New-Object System.Drawing.Size($Width, 24)
    $label.TextAlign = [System.Drawing.ContentAlignment]::MiddleLeft
    $Parent.Controls.Add($label)
    return $label
}

function Show-Message {
    param(
        [string]$Message,
        [string]$Title = 'Pickleball Reservation',
        [System.Windows.Forms.MessageBoxIcon]$Icon =
            [System.Windows.Forms.MessageBoxIcon]::Information
    )
    [System.Windows.Forms.MessageBox]::Show(
        $form,
        $Message,
        $Title,
        [System.Windows.Forms.MessageBoxButtons]::OK,
        $Icon
    ) | Out-Null
}

function Show-BookingSuccess {
    param([pscustomobject]$Details)

    [System.Media.SystemSounds]::Asterisk.Play()

    $dialog = New-Object System.Windows.Forms.Form
    $dialog.Text = 'Reservation confirmed'
    $dialog.StartPosition = 'CenterParent'
    $dialog.Size = New-Object System.Drawing.Size(600, 430)
    $dialog.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dialog.MaximizeBox = $false
    $dialog.MinimizeBox = $false
    $dialog.BackColor = [System.Drawing.Color]::FromArgb(232, 247, 238)
    $dialog.TopMost = $true

    $successTitle = New-Object System.Windows.Forms.Label
    $successTitle.Text = 'RESERVATION BOOKED!'
    $successTitle.Font = New-Object System.Drawing.Font(
        'Segoe UI Semibold',
        24
    )
    $successTitle.ForeColor = [System.Drawing.Color]::FromArgb(20, 120, 70)
    $successTitle.TextAlign = [System.Drawing.ContentAlignment]::MiddleCenter
    $successTitle.Location = New-Object System.Drawing.Point(25, 22)
    $successTitle.Size = New-Object System.Drawing.Size(535, 60)
    $dialog.Controls.Add($successTitle)

    $releaseSeconds = [double]$Details.releaseToConfirmationMs / 1000
    $apiSeconds = [double]$Details.bookingResponseMs / 1000
    $timingText = if (
        $Details.PSObject.Properties['releaseWasOpenAtStart'] -and
        $Details.releaseWasOpenAtStart
    ) {
        'Reservations were already open when this run started'
    } else {
        'Confirmed {0:N3} seconds after reservations opened' -f $releaseSeconds
    }
    $detailText =
        "Court: $($Details.court)`r`n" +
        "Date: $($Details.date)`r`n" +
        "Time: $($Details.startTime)-$($Details.endTime)`r`n" +
        "Additional player: $($Details.player)`r`n`r`n" +
        $timingText +
        "`r`n" +
        ('Booking API response: {0:N3} seconds' -f $apiSeconds) +
        "`r`n`r`n" +
        "Reservation: $($Details.reservation)"

    $successDetails = New-Object System.Windows.Forms.Label
    $successDetails.Text = $detailText
    $successDetails.Font = New-Object System.Drawing.Font('Segoe UI', 12)
    $successDetails.ForeColor = [System.Drawing.Color]::FromArgb(30, 60, 45)
    $successDetails.Location = New-Object System.Drawing.Point(55, 100)
    $successDetails.Size = New-Object System.Drawing.Size(485, 220)
    $successDetails.TextAlign = [System.Drawing.ContentAlignment]::TopLeft
    $dialog.Controls.Add($successDetails)

    $okButton = New-Object System.Windows.Forms.Button
    $okButton.Text = 'OK'
    $okButton.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 11)
    $okButton.Location = New-Object System.Drawing.Point(220, 330)
    $okButton.Size = New-Object System.Drawing.Size(150, 42)
    $okButton.BackColor = [System.Drawing.Color]::FromArgb(32, 126, 86)
    $okButton.ForeColor = [System.Drawing.Color]::White
    $okButton.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $okButton.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $dialog.AcceptButton = $okButton
    $dialog.Controls.Add($okButton)

    [void]$dialog.ShowDialog($form)
    $dialog.Dispose()
}

function Append-Status {
    param([string]$Text)
    $stamp = Get-Date -Format 'HH:mm:ss'
    $statusBox.AppendText("[$stamp] $Text`r`n")
    $statusBox.SelectionStart = $statusBox.TextLength
    $statusBox.ScrollToCaret()
    $form.Refresh()
}

function Process-ArmedOutputLine {
    param([string]$Text)

    $line = $Text.Trim()
    if (-not $line) {
        return
    }

    $successMarker = '__BOOKING_SUCCESS__'
    if ($line.StartsWith($successMarker)) {
        try {
            $script:SuccessDetails = $line.Substring(
                $successMarker.Length
            ) | ConvertFrom-Json
        } catch {
            Append-Status 'Booking succeeded, but confirmation details could not be parsed.'
        }
        return
    }

    $healthMarker = '__ARMED_HEALTH__'
    if ($line.StartsWith($healthMarker)) {
        try {
            $health = $line.Substring($healthMarker.Length) | ConvertFrom-Json
            $checkedAt = [DateTime]::Parse(
                [string]$health.checkedAt
            ).ToLocalTime()
            $releaseMinutes = [Math]::Ceiling(
                [double]$health.releaseInMs / 60000
            )
            Append-Status (
                'ARMED SESSION OK - authenticated API responded at ' +
                "$($checkedAt.ToString('h:mm:ss tt')); release in about " +
                "$releaseMinutes minute(s)."
            )
            $stateLabel.Text = 'ARMED - session OK'
            $stateLabel.ForeColor =
                [System.Drawing.Color]::FromArgb(20, 120, 70)
            $script:HealthCheckRequestedAt = $null
            $healthButton.Text = 'Check Armed Session'
            $healthButton.Enabled = $true
        } catch {
            Append-Status 'Authenticated health check returned unreadable details.'
        }
        return
    }

    $healthErrorMarker = '__ARMED_HEALTH_ERROR__'
    if ($line.StartsWith($healthErrorMarker)) {
        try {
            $healthError =
                $line.Substring($healthErrorMarker.Length) |
                    ConvertFrom-Json
            Append-Status (
                'ARMED SESSION CHECK FAILED - ' +
                "$($healthError.message) The process remains armed and will retry."
            )
            $stateLabel.Text = 'ARMED - check failed'
            $stateLabel.ForeColor = [System.Drawing.Color]::DarkOrange
            $script:HealthCheckRequestedAt = $null
            $healthButton.Text = 'Check Armed Session'
            $healthButton.Enabled = $true
            Show-Message `
                -Message (
                    "The health check failed, but the reservation remains armed.`r`n`r`n" +
                    "$($healthError.message)`r`n`r`n" +
                    'If repeated checks fail, stop the process, refresh login, test, and re-arm.'
                ) `
                -Title 'Armed session warning' `
                -Icon ([System.Windows.Forms.MessageBoxIcon]::Warning)
        } catch {
            Append-Status 'The armed health check failed with unreadable details.'
        }
        return
    }

    Append-Status $line
}

function Read-NewArmedOutput {
    param([switch]$Flush)

    if (Test-Path -LiteralPath $script:ArmedOutput) {
        $outputText = Get-Content -Raw -LiteralPath $script:ArmedOutput `
            -ErrorAction SilentlyContinue
        if ($null -eq $outputText) {
            $outputText = ''
        }

        if ($outputText.Length -lt $script:ArmedOutputOffset) {
            $script:ArmedOutputOffset = 0
            $script:ArmedOutputPending = ''
        }

        if ($outputText.Length -gt $script:ArmedOutputOffset) {
            $newText = $outputText.Substring($script:ArmedOutputOffset)
            $script:ArmedOutputOffset = $outputText.Length
            $combined = $script:ArmedOutputPending + $newText
            $parts = [regex]::Split($combined, '\r?\n')

            if ($combined -match '\r?\n$') {
                $script:ArmedOutputPending = ''
                $completeCount = $parts.Count
            } else {
                $script:ArmedOutputPending = $parts[$parts.Count - 1]
                $completeCount = $parts.Count - 1
            }

            for ($index = 0; $index -lt $completeCount; $index++) {
                Process-ArmedOutputLine -Text $parts[$index]
            }
        }
    }

    if ($Flush -and $script:ArmedOutputPending) {
        Process-ArmedOutputLine -Text $script:ArmedOutputPending
        $script:ArmedOutputPending = ''
    }
}

function Get-TimeOptions {
    $items = New-Object System.Collections.Generic.List[string]
    for ($hour = 0; $hour -lt 24; $hour++) {
        $items.Add(('{0:D2}:00' -f $hour))
        $items.Add(('{0:D2}:30' -f $hour))
    }
    return $items
}

function Update-ReleaseLabel {
    $target = $datePicker.Value.Date
    $release = $target.AddDays(-10).AddHours(7)
    $releaseLabel.Text = 'Expected release: ' +
        $release.ToString('dddd, MMMM d, yyyy ''at'' h:mm:ss tt')
}

function Save-GuiConfig {
    $start = [string]$startCombo.SelectedItem
    $end = [string]$endCombo.SelectedItem
    if ([string]::IsNullOrWhiteSpace($start) -or
        [string]::IsNullOrWhiteSpace($end)) {
        throw 'Choose both a start and end time.'
    }
    $startValue = [TimeSpan]::ParseExact($start, 'hh\:mm', $null)
    $endValue = [TimeSpan]::ParseExact($end, 'hh\:mm', $null)
    if ($endValue -le $startValue) {
        throw 'End time must be after start time.'
    }
    $player = $playerCombo.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($player)) {
        throw 'Enter the other player''s exact PlayByPoint name or Guest Player.'
    }

    $config = Load-ReservationConfig
    $config.targetDate = $datePicker.Value.ToString('yyyy-MM-dd')
    $config.startTime = $start
    $config.endTime = $end
    $flexibilityMinutes =
        if ($flexibilityCombo.SelectedIndex -eq 1) { 30 } else { 0 }
    if ($config.PSObject.Properties['timeFlexibilityMinutes']) {
        $config.timeFlexibilityMinutes = $flexibilityMinutes
    } else {
        $config | Add-Member `
            -NotePropertyName 'timeFlexibilityMinutes' `
            -NotePropertyValue $flexibilityMinutes
    }
    $config.partySize = 2
    $config.additionalPlayers = @($player)

    $selectedCourt = [string]$courtCombo.SelectedItem
    if ($selectedCourt -and $selectedCourt -ne 'Any available court') {
        $remaining = @(
            $config.courtPreferences |
                Where-Object { $_ -ne $selectedCourt }
        )
        $config.courtPreferences = @($selectedCourt) + $remaining
    }

    $json = $config | ConvertTo-Json -Depth 20
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($configPath, $json, $utf8WithoutBom)
    Append-Status (
        "Saved: $($config.targetDate), $start-$end, with $player, " +
        "$selectedCourt, $($flexibilityCombo.SelectedItem)."
    )
}

function Invoke-NodeCapture {
    param([string[]]$Arguments)
    $quoted = $Arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $nodePath
    $startInfo.Arguments = $quoted -join ' '
    $startInfo.WorkingDirectory = $projectDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output = $stdout
        Error = $stderr
    }
}

$config = Load-ReservationConfig

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Pickleball Reservation'
$form.StartPosition = 'CenterScreen'
$form.Size = New-Object System.Drawing.Size(660, 710)
$form.MinimumSize = New-Object System.Drawing.Size(660, 710)
$form.MaximizeBox = $false
$form.Font = New-Object System.Drawing.Font('Segoe UI', 10)
$form.BackColor = [System.Drawing.Color]::FromArgb(246, 248, 251)

$title = New-Object System.Windows.Forms.Label
$title.Text = 'PlayByPoint Pickleball Reservation'
$title.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 18)
$title.Location = New-Object System.Drawing.Point(24, 18)
$title.Size = New-Object System.Drawing.Size(590, 38)
$form.Controls.Add($title)

$subtitle = New-Object System.Windows.Forms.Label
$subtitle.Text = 'Configure the reservation, test the API, then arm it for release.'
$subtitle.ForeColor = [System.Drawing.Color]::DimGray
$subtitle.Location = New-Object System.Drawing.Point(27, 57)
$subtitle.Size = New-Object System.Drawing.Size(580, 25)
$form.Controls.Add($subtitle)

Add-Label -Parent $form -Text 'Reservation date' -X 28 -Y 103 | Out-Null
$datePicker = New-Object System.Windows.Forms.DateTimePicker
$datePicker.Format = [System.Windows.Forms.DateTimePickerFormat]::Custom
$datePicker.CustomFormat = 'dddd, MMMM d, yyyy'
$datePicker.Location = New-Object System.Drawing.Point(180, 103)
$datePicker.Size = New-Object System.Drawing.Size(360, 28)
if ($config.targetDate) {
    $datePicker.Value = [DateTime]::ParseExact(
        [string]$config.targetDate,
        'yyyy-MM-dd',
        [Globalization.CultureInfo]::InvariantCulture
    )
}
$form.Controls.Add($datePicker)

$releaseLabel = New-Object System.Windows.Forms.Label
$releaseLabel.ForeColor = [System.Drawing.Color]::FromArgb(70, 90, 120)
$releaseLabel.Location = New-Object System.Drawing.Point(180, 134)
$releaseLabel.Size = New-Object System.Drawing.Size(430, 25)
$form.Controls.Add($releaseLabel)

Add-Label -Parent $form -Text 'Start time' -X 28 -Y 172 | Out-Null
$startCombo = New-Object System.Windows.Forms.ComboBox
$startCombo.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
$startCombo.Location = New-Object System.Drawing.Point(180, 172)
$startCombo.Size = New-Object System.Drawing.Size(150, 28)
[void]$startCombo.Items.AddRange([object[]](Get-TimeOptions))
$startCombo.SelectedItem = [string]$config.startTime
$form.Controls.Add($startCombo)

Add-Label -Parent $form -Text 'End time' -X 350 -Y 172 -Width 90 | Out-Null
$endCombo = New-Object System.Windows.Forms.ComboBox
$endCombo.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
$endCombo.Location = New-Object System.Drawing.Point(440, 172)
$endCombo.Size = New-Object System.Drawing.Size(150, 28)
[void]$endCombo.Items.AddRange([object[]](Get-TimeOptions))
$endCombo.SelectedItem = [string]$config.endTime
$form.Controls.Add($endCombo)

Add-Label -Parent $form -Text 'Additional player' -X 28 -Y 218 | Out-Null
$playerCombo = New-Object System.Windows.Forms.ComboBox
$playerCombo.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDown
$playerCombo.AutoCompleteMode = [System.Windows.Forms.AutoCompleteMode]::SuggestAppend
$playerCombo.AutoCompleteSource = [System.Windows.Forms.AutoCompleteSource]::ListItems
$playerCombo.Location = New-Object System.Drawing.Point(180, 218)
$playerCombo.Size = New-Object System.Drawing.Size(410, 28)
$savedPlayer = [string]$config.additionalPlayers[0]
$knownPlayers = @('Bing Kung', 'Nina Tran', 'grand lo')
foreach ($knownPlayer in $knownPlayers) {
    [void]$playerCombo.Items.Add($knownPlayer)
}
if ($savedPlayer -and $knownPlayers -notcontains $savedPlayer) {
    [void]$playerCombo.Items.Add($savedPlayer)
}
$playerCombo.Text = $savedPlayer
$form.Controls.Add($playerCombo)

$playerHint = New-Object System.Windows.Forms.Label
$playerHint.Text = 'Use the exact PlayByPoint display name, or Guest Player.'
$playerHint.ForeColor = [System.Drawing.Color]::DimGray
$playerHint.Location = New-Object System.Drawing.Point(180, 247)
$playerHint.Size = New-Object System.Drawing.Size(410, 24)
$form.Controls.Add($playerHint)

Add-Label -Parent $form -Text 'Court preference' -X 28 -Y 281 | Out-Null
$courtCombo = New-Object System.Windows.Forms.ComboBox
$courtCombo.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
$courtCombo.Location = New-Object System.Drawing.Point(180, 281)
$courtCombo.Size = New-Object System.Drawing.Size(250, 28)
[void]$courtCombo.Items.Add('Any available court')
foreach ($court in $config.courtPreferences) {
    [void]$courtCombo.Items.Add([string]$court)
}
$courtCombo.SelectedIndex = 0
$form.Controls.Add($courtCombo)

Add-Label -Parent $form -Text 'Time flexibility' -X 28 -Y 324 | Out-Null
$flexibilityCombo = New-Object System.Windows.Forms.ComboBox
$flexibilityCombo.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
$flexibilityCombo.Location = New-Object System.Drawing.Point(180, 324)
$flexibilityCombo.Size = New-Object System.Drawing.Size(300, 28)
[void]$flexibilityCombo.Items.Add('Exact time only')
[void]$flexibilityCombo.Items.Add('Allow 30 minutes earlier or later')
$flexibilityCombo.SelectedIndex =
    if ($null -ne $config.timeFlexibilityMinutes -and
        [int]$config.timeFlexibilityMinutes -eq 30) {
        1
    } else {
        0
    }
$form.Controls.Add($flexibilityCombo)

$loginButton = New-Object System.Windows.Forms.Button
$loginButton.Text = '1. Login / Refresh Session'
$loginButton.Location = New-Object System.Drawing.Point(28, 375)
$loginButton.Size = New-Object System.Drawing.Size(185, 42)
$loginButton.BackColor = [System.Drawing.Color]::White
$form.Controls.Add($loginButton)

$testButton = New-Object System.Windows.Forms.Button
$testButton.Text = '2. Save + Test API'
$testButton.Location = New-Object System.Drawing.Point(226, 375)
$testButton.Size = New-Object System.Drawing.Size(175, 42)
$testButton.BackColor = [System.Drawing.Color]::White
$form.Controls.Add($testButton)

$armButton = New-Object System.Windows.Forms.Button
$armButton.Text = '3. Arm Reservation'
$armButton.Location = New-Object System.Drawing.Point(414, 375)
$armButton.Size = New-Object System.Drawing.Size(176, 42)
$armButton.BackColor = [System.Drawing.Color]::FromArgb(32, 126, 86)
$armButton.ForeColor = [System.Drawing.Color]::White
$armButton.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
$form.Controls.Add($armButton)

$stopButton = New-Object System.Windows.Forms.Button
$stopButton.Text = 'Stop Armed Process'
$stopButton.Location = New-Object System.Drawing.Point(414, 426)
$stopButton.Size = New-Object System.Drawing.Size(176, 34)
$stopButton.Enabled = $false
$form.Controls.Add($stopButton)

$healthButton = New-Object System.Windows.Forms.Button
$healthButton.Text = 'Check Armed Session'
$healthButton.Location = New-Object System.Drawing.Point(226, 426)
$healthButton.Size = New-Object System.Drawing.Size(175, 34)
$healthButton.Enabled = $false
$form.Controls.Add($healthButton)

$stateLabel = New-Object System.Windows.Forms.Label
$stateLabel.Text = 'Not armed'
$stateLabel.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 11)
$stateLabel.ForeColor = [System.Drawing.Color]::DarkRed
$stateLabel.Location = New-Object System.Drawing.Point(28, 431)
$stateLabel.Size = New-Object System.Drawing.Size(185, 28)
$form.Controls.Add($stateLabel)

$statusBox = New-Object System.Windows.Forms.TextBox
$statusBox.Multiline = $true
$statusBox.ReadOnly = $true
$statusBox.ScrollBars = [System.Windows.Forms.ScrollBars]::Vertical
$statusBox.BackColor = [System.Drawing.Color]::White
$statusBox.Font = New-Object System.Drawing.Font('Consolas', 9)
$statusBox.Location = New-Object System.Drawing.Point(28, 476)
$statusBox.Size = New-Object System.Drawing.Size(562, 145)
$form.Controls.Add($statusBox)

$footer = New-Object System.Windows.Forms.Label
$footer.Text = 'Keep this application and its dedicated Chrome window open after arming.'
$footer.ForeColor = [System.Drawing.Color]::DimGray
$footer.Location = New-Object System.Drawing.Point(28, 627)
$footer.Size = New-Object System.Drawing.Size(562, 24)
$form.Controls.Add($footer)

$datePicker.Add_ValueChanged({ Update-ReleaseLabel })
$datePicker.Add_ValueChanged({ $script:PreflightPassed = $false })
$startCombo.Add_SelectedIndexChanged({ $script:PreflightPassed = $false })
$endCombo.Add_SelectedIndexChanged({ $script:PreflightPassed = $false })
$playerCombo.Add_TextChanged({ $script:PreflightPassed = $false })
$playerCombo.Add_SelectedIndexChanged({ $script:PreflightPassed = $false })
$courtCombo.Add_SelectedIndexChanged({ $script:PreflightPassed = $false })
$flexibilityCombo.Add_SelectedIndexChanged({ $script:PreflightPassed = $false })

$loginButton.Add_Click({
    if ($script:ArmedProcess -and -not $script:ArmedProcess.HasExited) {
        Show-Message 'Use Check Armed Session while the reservation is armed.'
        return
    }
    if ($script:LoginProcess -and -not $script:LoginProcess.HasExited) {
        Show-Message 'The login Chrome window is already open.'
        return
    }
    Append-Status 'Opening ordinary Chrome with the app''s persistent profile.'
    Append-Status 'Sign in, reach PlayByPoint Home, then close that Chrome window.'
    $script:PreflightPassed = $false
    $script:LoginProcess = Start-Process `
        -FilePath $nodePath `
        -ArgumentList ('"{0}"' -f $loginScript) `
        -WorkingDirectory $projectDirectory `
        -WindowStyle Hidden `
        -PassThru
    $loginButton.Enabled = $false
})

$testButton.Add_Click({
    try {
        if ($script:ArmedProcess -and -not $script:ArmedProcess.HasExited) {
            throw (
                'Save + Test API cannot run while armed because the live ' +
                'process owns the Chrome profile. Use Check Armed Session instead.'
            )
        }
        if ($script:LoginProcess -and -not $script:LoginProcess.HasExited) {
            throw 'Finish login and close the dedicated Chrome window first.'
        }
        Save-GuiConfig
        Append-Status 'Running authenticated read-only API preflight...'
        $testButton.Enabled = $false
        $armButton.Enabled = $false
        $form.Refresh()
        $result = Invoke-NodeCapture @($apiScript, '--preflight')
        if ($result.Output) {
            $result.Output.Trim().Split("`n") |
                ForEach-Object { Append-Status $_.Trim() }
        }
        if ($result.ExitCode -ne 0) {
            throw ($result.Error.Trim() + "`r`n" + $result.Output.Trim())
        }
        Append-Status 'PREFLIGHT PASSED. No reservation was created.'
        $script:PreflightPassed = $true
        Show-Message 'API preflight passed. No reservation was created.'
    } catch {
        $script:PreflightPassed = $false
        Append-Status "PREFLIGHT FAILED: $($_.Exception.Message)"
        Show-Message `
            -Message $_.Exception.Message `
            -Title 'Preflight failed' `
            -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
    } finally {
        $isArmed =
            $script:ArmedProcess -and -not $script:ArmedProcess.HasExited
        $testButton.Enabled = -not $isArmed
        if (-not $isArmed) {
            $armButton.Enabled = $true
        }
    }
})

$armButton.Add_Click({
    try {
        if ($script:LoginProcess -and -not $script:LoginProcess.HasExited) {
            throw 'Finish login and close the dedicated Chrome window first.'
        }
        if ($script:ArmedProcess -and -not $script:ArmedProcess.HasExited) {
            throw 'A reservation process is already armed.'
        }
        if (-not $script:PreflightPassed) {
            throw 'Run Save + Test API successfully before arming.'
        }
        Save-GuiConfig
        $summary =
            "Arm this live reservation?`r`n`r`n" +
            "Date: $($datePicker.Value.ToString('dddd, MMMM d, yyyy'))`r`n" +
            "Time: $($startCombo.SelectedItem)-$($endCombo.SelectedItem)`r`n" +
            "Flexibility: $($flexibilityCombo.SelectedItem)`r`n" +
            "Player: $($playerCombo.Text.Trim())`r`n" +
            "Court: $($courtCombo.SelectedItem)`r`n`r`n" +
            'At release, the app will send the live reservation POST.'
        $answer = [System.Windows.Forms.MessageBox]::Show(
            $form,
            $summary,
            'Confirm live reservation',
            [System.Windows.Forms.MessageBoxButtons]::YesNo,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        )
        if ($answer -ne [System.Windows.Forms.DialogResult]::Yes) {
            Append-Status 'Arm cancelled.'
            return
        }

        Remove-Item -LiteralPath $script:ArmedOutput -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $script:ArmedError -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $script:HealthCheckRequest -Force -ErrorAction SilentlyContinue
        $script:ArmedOutputOffset = 0
        $script:ArmedOutputPending = ''
        $script:SuccessDetails = $null
        $script:HealthCheckRequestedAt = $null
        $script:ArmedProcess = Start-Process `
            -FilePath $nodePath `
            -ArgumentList ('"{0}" --live' -f $apiScript) `
            -WorkingDirectory $projectDirectory `
            -RedirectStandardOutput $script:ArmedOutput `
            -RedirectStandardError $script:ArmedError `
            -WindowStyle Hidden `
            -PassThru
        $stateLabel.Text = "ARMED - process $($script:ArmedProcess.Id)"
        $stateLabel.ForeColor = [System.Drawing.Color]::FromArgb(20, 120, 70)
        $armButton.Enabled = $false
        $testButton.Enabled = $false
        $loginButton.Enabled = $false
        $healthButton.Enabled = $true
        $stopButton.Enabled = $true
        Append-Status "LIVE reservation armed as process $($script:ArmedProcess.Id)."
        Append-Status 'Keep this GUI and the dedicated Chrome window open.'
    } catch {
        Append-Status "ARM FAILED: $($_.Exception.Message)"
        Show-Message `
            -Message $_.Exception.Message `
            -Title 'Unable to arm' `
            -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
    }
})

$healthButton.Add_Click({
    try {
        if (-not $script:ArmedProcess -or $script:ArmedProcess.HasExited) {
            throw 'No live reservation process is currently armed.'
        }
        $request = [pscustomobject]@{
            requestedAt = (Get-Date).ToUniversalTime().ToString('o')
            processId = $script:ArmedProcess.Id
        }
        $json = $request | ConvertTo-Json
        $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText(
            $script:HealthCheckRequest,
            $json,
            $utf8WithoutBom
        )
        $script:HealthCheckRequestedAt = Get-Date
        $healthButton.Enabled = $false
        $healthButton.Text = 'Checking...'
        Append-Status 'Requested an authenticated health check from the armed process.'
    } catch {
        Append-Status "HEALTH CHECK FAILED: $($_.Exception.Message)"
        Show-Message `
            -Message $_.Exception.Message `
            -Title 'Unable to check armed session' `
            -Icon ([System.Windows.Forms.MessageBoxIcon]::Error)
    }
})

$stopButton.Add_Click({
    if ($script:ArmedProcess -and -not $script:ArmedProcess.HasExited) {
        $answer = [System.Windows.Forms.MessageBox]::Show(
            $form,
            'Stop the armed reservation process?',
            'Confirm stop',
            [System.Windows.Forms.MessageBoxButtons]::YesNo,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        )
        if ($answer -eq [System.Windows.Forms.DialogResult]::Yes) {
            Stop-Process -Id $script:ArmedProcess.Id -Force -ErrorAction SilentlyContinue
            Append-Status 'Armed process stopped.'
        }
    }
})

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 1500
$timer.Add_Tick({
    if ($script:LoginProcess -and $script:LoginProcess.HasExited) {
        $loginButton.Enabled = $true
        if ($script:LoginProcess.ExitCode -eq 0) {
            Append-Status 'Login Chrome closed; persistent profile saved.'
        } else {
            Append-Status "Login process exited with code $($script:LoginProcess.ExitCode)."
        }
        $script:LoginProcess = $null
    }

    if ($script:ArmedProcess) {
        Read-NewArmedOutput
        if ($script:ArmedProcess.HasExited) {
            Read-NewArmedOutput -Flush
            $errorText = Get-Content -Raw -LiteralPath $script:ArmedError -ErrorAction SilentlyContinue
            if ($errorText) {
                Append-Status $errorText.Trim()
            }
            if (-not $script:SuccessDetails -and
                (Test-Path -LiteralPath $script:ArmedOutput)) {
                $finalOutput = Get-Content -Raw -LiteralPath $script:ArmedOutput
                $markerMatch = [regex]::Match(
                    $finalOutput,
                    '(?m)^__BOOKING_SUCCESS__(.+)$'
                )
                if ($markerMatch.Success) {
                    try {
                        $script:SuccessDetails =
                            $markerMatch.Groups[1].Value.Trim() |
                                ConvertFrom-Json
                    } catch {
                        Append-Status 'Booking succeeded, but confirmation details could not be parsed.'
                    }
                }
            }
            if ($script:ArmedProcess.ExitCode -eq 0) {
                $stateLabel.Text = 'Completed'
                $stateLabel.ForeColor = [System.Drawing.Color]::FromArgb(20, 120, 70)
                if ($script:SuccessDetails) {
                    Show-BookingSuccess -Details $script:SuccessDetails
                }
            } else {
                $stateLabel.Text = "Stopped with error $($script:ArmedProcess.ExitCode)"
                $stateLabel.ForeColor = [System.Drawing.Color]::DarkRed
            }
            $armButton.Enabled = $true
            $testButton.Enabled = $true
            $loginButton.Enabled = $true
            $healthButton.Enabled = $false
            $healthButton.Text = 'Check Armed Session'
            $stopButton.Enabled = $false
            $script:ArmedProcess = $null
            $script:ArmedOutputOffset = 0
            $script:ArmedOutputPending = ''
            $script:SuccessDetails = $null
            $script:HealthCheckRequestedAt = $null
        }
    }

    if ($script:HealthCheckRequestedAt -and
        ((Get-Date) - $script:HealthCheckRequestedAt).TotalSeconds -gt 15) {
        Append-Status (
            'Health check has not answered within 15 seconds. The armed process ' +
            'may be entering its final prewarm window; review the live status.'
        )
        $script:HealthCheckRequestedAt = $null
        if ($script:ArmedProcess -and -not $script:ArmedProcess.HasExited) {
            $healthButton.Enabled = $true
            $healthButton.Text = 'Check Armed Session'
        }
    }
})
$timer.Start()

$form.Add_FormClosing({
    if ($script:ArmedProcess -and -not $script:ArmedProcess.HasExited) {
        $answer = [System.Windows.Forms.MessageBox]::Show(
            $form,
            'A live reservation is armed. Closing the GUI will leave it running in the background. Close anyway?',
            'Reservation is armed',
            [System.Windows.Forms.MessageBoxButtons]::YesNo,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        )
        if ($answer -ne [System.Windows.Forms.DialogResult]::Yes) {
            $_.Cancel = $true
        }
    }
})

Update-ReleaseLabel
Append-Status 'Ready. Start with Login / Refresh Session.'
[void]$form.ShowDialog()
