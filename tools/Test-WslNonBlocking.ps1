param(
    [string]$Distro = "Ubuntu",
    [string]$JobDirWsl = "/tmp/wsl_nonblocking_probe_job",
    [switch]$Full,
    [int]$LaunchPassThresholdMs = 3000,
    [int]$ObserveSeconds = 20,
    [switch]$WaitForCompletion
)

$ErrorActionPreference = "Stop"

function Get-UtcIso {
    return [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

$repoWsl = "/home/dev/workspace/lrc_chunker"
$repoUnc = "\\wsl.localhost\$Distro\home\dev\workspace\lrc_chunker"
$jobDirUnc = "\\wsl.localhost\$Distro" + ($JobDirWsl -replace '/', '\')
$inputDirUnc = Join-Path $jobDirUnc "input"
$outputDirUnc = Join-Path $jobDirUnc "output"
$statusUnc = Join-Path $jobDirUnc "status.json"
$stdoutUnc = Join-Path $jobDirUnc "stdout.log"
$stderrUnc = Join-Path $jobDirUnc "stderr.log"
$resultUnc = Join-Path $outputDirUnc "result.lrc"

$testLrcUnc = Join-Path $repoUnc "test\Cry Cry Cry - Coldplay.lrc"
$testAudioUnc = Join-Path $repoUnc "test\Cry Cry Cry - Coldplay.wav"
$launchScriptWsl = "$repoWsl/tools/ae_launch_wsl.sh"
$jobInputLrcWsl = "$JobDirWsl/input/song.lrc"
$jobInputAudioWsl = "$JobDirWsl/input/song.wav"
$jobOutputLrcWsl = "$JobDirWsl/output/result.lrc"
$jobStatusWsl = "$JobDirWsl/status.json"
$jobCancelWsl = "$JobDirWsl/cancel.flag"
$requestUnc = Join-Path $jobDirUnc "request.json"

if (-not (Test-Path -LiteralPath $testLrcUnc)) {
    throw "Missing test LRC: $testLrcUnc"
}
if (-not (Test-Path -LiteralPath $testAudioUnc)) {
    throw "Missing test audio: $testAudioUnc"
}

if (Test-Path -LiteralPath $jobDirUnc) {
    Remove-Item -LiteralPath $jobDirUnc -Recurse -Force
}
New-Item -ItemType Directory -Path $inputDirUnc | Out-Null
New-Item -ItemType Directory -Path $outputDirUnc | Out-Null

Copy-Item -LiteralPath $testLrcUnc -Destination (Join-Path $inputDirUnc "song.lrc")
Copy-Item -LiteralPath $testAudioUnc -Destination (Join-Path $inputDirUnc "song.wav")

$options = [ordered]@{
    model = "medium.en"
    language = "en"
    profile = "slow_attack"
    use_lrc_anchors = $true
}
if ($Full) {
    $options["denoiser"] = "auto"
} else {
    $options["denoiser"] = "none"
    $options["alignment_backend"] = "lrc"
    $options["use_lrc_anchors"] = $false
}

$request = [ordered]@{
    protocol_version = 1
    job_id = [IO.Path]::GetFileName($JobDirWsl)
    created_utc = Get-UtcIso
    input = [ordered]@{
        mode = "single"
        lrc_path = $jobInputLrcWsl
        audio_path = $jobInputAudioWsl
    }
    output = [ordered]@{
        result_lrc_path = $jobOutputLrcWsl
    }
    options = $options
    callback = [ordered]@{
        status_file = $jobStatusWsl
        cancel_flag = $jobCancelWsl
    }
}

$request | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $requestUnc -Encoding UTF8

$launchArgs = @(
    "-d", $Distro,
    "bash", "-lc",
    "$launchScriptWsl $JobDirWsl"
)

$sw = [Diagnostics.Stopwatch]::StartNew()
$launchOutput = & wsl.exe @launchArgs 2>&1
$launchExitCode = $LASTEXITCODE
$sw.Stop()
$launchMs = [int]$sw.ElapsedMilliseconds

$timeline = @()
$status = $null
$runningSeen = $false
$completedSeen = $false
$deadline = (Get-Date).AddSeconds($ObserveSeconds)

while ((Get-Date) -lt $deadline) {
    $status = Read-JsonFile -Path $statusUnc
    if ($null -ne $status) {
        $timeline += [ordered]@{
            local_time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            state = [string]$status.state
            stage = [string]$status.stage
            heartbeat_utc = [string]$status.heartbeat_utc
        }
        if ([string]$status.state -eq "running") {
            $runningSeen = $true
        }
        if (@("completed","failed","cancelled") -contains [string]$status.state) {
            $completedSeen = $true
            break
        }
    }
    Start-Sleep -Milliseconds 1000
}

if ($WaitForCompletion -and -not $completedSeen) {
    while ($true) {
        $status = Read-JsonFile -Path $statusUnc
        if ($null -ne $status -and @("completed","failed","cancelled") -contains [string]$status.state) {
            $completedSeen = $true
            break
        }
        Start-Sleep -Milliseconds 1000
    }
}

$status = Read-JsonFile -Path $statusUnc
$resultExists = Test-Path -LiteralPath $resultUnc
$nonBlockingPass = ($launchExitCode -eq 0 -and $launchMs -lt $LaunchPassThresholdMs)
$backgroundProgressPass = ($null -ne $status -and ($runningSeen -or $completedSeen))
$overallPass = ($nonBlockingPass -and $backgroundProgressPass)
$testKind = if ($Full) { "non_blocking_full" } else { "non_blocking_quick" }
$finalState = if ($null -ne $status) { [string]$status.state } else { "" }
$finalStage = if ($null -ne $status) { [string]$status.stage } else { "" }

$summary = [ordered]@{
    ok = $overallPass
    test_kind = $testKind
    distro = $Distro
    job_dir_wsl = $JobDirWsl
    job_dir_unc = $jobDirUnc
    launch_exit_code = $launchExitCode
    launch_elapsed_ms = $launchMs
    launch_pass_threshold_ms = $LaunchPassThresholdMs
    non_blocking_launch_pass = $nonBlockingPass
    background_progress_pass = $backgroundProgressPass
    running_seen = $runningSeen
    terminal_seen_during_observe = $completedSeen
    observed_seconds = $ObserveSeconds
    wait_for_completion = [bool]$WaitForCompletion
    final_state = $finalState
    final_stage = $finalStage
    result_lrc_exists = $resultExists
    status_path_unc = $statusUnc
    result_lrc_path_unc = $resultUnc
    stdout_log_unc = $stdoutUnc
    stderr_log_unc = $stderrUnc
    launch_output = [string]($launchOutput | Out-String).Trim()
    timeline = $timeline
}

if ($null -ne $status) {
    $summary["status"] = $status
}

$summary | ConvertTo-Json -Depth 10
