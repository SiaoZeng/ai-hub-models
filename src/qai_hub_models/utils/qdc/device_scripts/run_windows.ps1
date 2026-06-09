# genie-t2t-run.exe writes UTF-8 to stdout, and if we don't configure the
# powershell terminal correctly we'll end up capturing mojibake.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

# genie-t2t-run.exe fails randomly on QDC devices; give each invocation one
# retry before letting the failure abort the job. The .exe is run via
# assignment (not piped) so $LASTEXITCODE reflects it rather than Out-File,
# which always exits 0 and would defeat the retry. PowerShell has no `set -e`,
# so we throw on a double-failure to fail the job (the captured output is also
# only committed to $OutFile on success, keeping a crashed attempt out of it).
function Invoke-GenieRetry {
    param(
        [Parameter(Mandatory = $true)][string[]]$GenieArgs,
        [string]$OutFile
    )
    foreach ($attempt in 1, 2) {
        $captured = & genie-t2t-run.exe @GenieArgs 2>&1
        if ($LASTEXITCODE -eq 0) {
            if ($OutFile) { $captured | Out-File -FilePath $OutFile -Append -Encoding utf8 }
            return
        }
        Write-Host "Invoke-GenieRetry: genie-t2t-run.exe failed (exit $LASTEXITCODE)"
    }
    throw "genie-t2t-run.exe failed twice: genie-t2t-run.exe $($GenieArgs -join ' ')"
}

Set-Location C:\Temp\TestContent\

# Verify network connectivity before download
Write-Host "=== Pre-download connectivity check ==="
Write-Host "Pinging google.com before QAIRT SDK download..."
$prePing = Test-Connection -ComputerName google.com -Count 1 -Quiet
if ($prePing) { Write-Host "Pre-download ping: SUCCESS" } else { Write-Host "Pre-download ping: FAILED" }

$source = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip"
$output = "C:\Temp\TestContent\qairt.zip"
(New-Object System.Net.WebClient).DownloadFile($source, $output)

# Verify network connectivity after download
Write-Host "=== Post-download connectivity check ==="
Write-Host "Pinging google.com after QAIRT SDK download..."
$postPing = Test-Connection -ComputerName google.com -Count 1 -Quiet
if ($postPing) { Write-Host "Post-download ping: SUCCESS (WiFi active)" } else { Write-Host "Post-download ping: FAILED (WiFi down)" }
Expand-Archive -Path "C:\Temp\TestContent\qairt.zip" -DestinationPath "C:\Temp\TestContent\"
$env:QAIRT_HOME = "C:\Temp\TestContent\qairt\{QAIRT_VERSION}"
$env:Path = "$env:QAIRT_HOME\bin\aarch64-windows-msvc;" + $env:Path
$env:Path = "$env:QAIRT_HOME\lib\aarch64-windows-msvc;" + $env:Path
$env:ADSP_LIBRARY_PATH = "$env:QAIRT_HOME\lib\hexagon-{HEXAGON_VERSION}\unsigned"

Invoke-GenieRetry -GenieArgs @("-c", "genie_config.json", "--prompt_file", "sample_prompt.txt") -OutFile "C:/Temp/QDC_logs/genie.log"

for ($i = 1; $i -le {NUM_TRIALS}; $i++) {
    $profileName = "profile$($i).txt"
    $outputPath = "C:/Temp/QDC_logs/$profileName"
    (Get-Content genie_config.json) -replace '"seed": \d+', "`"seed`": $i" | Set-Content genie_config.json
    Invoke-GenieRetry -GenieArgs @("-c", "genie_config.json", "--prompt_file", "sample_prompt.txt", "--profile", $outputPath)
}

$PromptDir = "C:\Temp\TestContent\prompts"
$EvalOutputFile = "C:/Temp/QDC_logs/eval_outputs.txt"
if (Test-Path $PromptDir) {
    New-Item -ItemType Directory -Force -Path "C:/Temp/QDC_logs"
    "" | Out-File -FilePath $EvalOutputFile -Encoding utf8
    $promptFiles = Get-ChildItem -Path $PromptDir -Filter "prompt_*.txt" | Sort-Object Name
    foreach ($promptFile in $promptFiles) {
        $idx = [regex]::Match($promptFile.Name, 'prompt_(\d+)\.txt').Groups[1].Value
        "===EVAL_IDX_${idx}===" | Out-File -FilePath $EvalOutputFile -Append -Encoding utf8
        Invoke-GenieRetry -GenieArgs @("-c", "genie_config.json", "--prompt_file", $promptFile.FullName) -OutFile $EvalOutputFile
    }
}
