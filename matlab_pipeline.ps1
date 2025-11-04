param(
    [string]$WatchPath = "",
    [string]$TestJobPath = ""
)

function Invoke-MatlabScript {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ScriptName,
        
        [Parameter(Mandatory=$false)]
        [array]$Parameters = @(),
        
        [string]$LogFile = "",
        [string]$WorkingDirectory = "",
        [int]$TimeoutMinutes = 30,
        [switch]$ShowOutput,
        [switch]$AddPath
    )
    
    # Genera nome log se non specificato
    if ([string]::IsNullOrEmpty($LogFile)) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $LogFile = "${ScriptName}_${timestamp}.log"
    }
    
    # Working directory
    $originalLocation = Get-Location
    if (-not [string]::IsNullOrEmpty($WorkingDirectory)) {
        Set-Location $WorkingDirectory
    }
    
    try {
        # Costruisci la lista dei parametri per MATLAB
        $paramString = ""
        if ($Parameters.Count -gt 0) {
            $quotedParams = $Parameters | ForEach-Object { 
                if ($_ -is [string]) {
                    "'$($_.Replace("'", "''"))'"  # Escape single quotes
                } else {
                    $_.ToString()
                }
            }
            $paramString = $quotedParams -join ", "
        }
        
        # Costruisci il comando MATLAB
        $matlabCommands = @()
        
        if ($AddPath) {
            $matlabCommands += "addpath(genpath(pwd))"
        }
        
        # Comando principale
        if ($Parameters.Count -gt 0) {
            $matlabCommands += "${ScriptName}(${paramString})"
        } else {
            $matlabCommands += $ScriptName
        }
        
        # Script MATLAB completo
        $matlabScript = $matlabCommands -join "; "
        
        # Costruisci il comando completo come stringa
        $fullCommand = "matlab -batch `"$matlabScript`""
        
        # Log header
        $logHeader = @"
===============================================
MATLAB Script Execution Log
===============================================
Script: $ScriptName
Parameters: [$($Parameters -join ', ')]
Working Directory: $(Get-Location)
Started: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Timeout: $TimeoutMinutes minutes
===============================================

Full Command:
$fullCommand

MATLAB Script:
$matlabScript

Output:
===============================================

"@
        
        $logHeader | Out-File -FilePath $LogFile -Encoding UTF8
        
        Write-Host "Executing MATLAB script: $ScriptName" -ForegroundColor Cyan
        Write-Host "Parameters: [$($Parameters -join ', ')]" -ForegroundColor Gray
        Write-Host "Full command: $fullCommand" -ForegroundColor Yellow
        Write-Host "Log file: $LogFile" -ForegroundColor Gray
        
        # Esegui MATLAB nella shell corrente con cattura output
        $startTime = Get-Date
        $output = ""
        $errorOutput = ""
        $exitCode = 0
        
        try {
            # Esegui il comando e cattura tutto l'output in modo interattivo
            Write-Host "Executing command in current shell..." -ForegroundColor Yellow
            
            # Usa Tee-Object per mostrare output in tempo reale E salvare su file
            $tempLogFile = "$LogFile.temp"
            
            # Esegui MATLAB con output interattivo e logging simultaneo
            & matlab -batch $matlabScript 2>&1 | Tee-Object -FilePath $tempLogFile
            $exitCode = $LASTEXITCODE
            
            # Leggi tutto l'output dal file temporaneo
            $output = if (Test-Path $tempLogFile) {
                Get-Content $tempLogFile -Raw
            } else {
                ""
            }
            
            # Pulisci il file temporaneo
            Remove-Item $tempLogFile -ErrorAction SilentlyContinue
            
            if ($null -eq $exitCode) {
                $exitCode = 0
            }
            
            Write-Host "Command completed with exit code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })
        }
        catch {
            $errorOutput = $_.Exception.Message
            $exitCode = 1
            Write-Host "Command failed with exception: $errorOutput" -ForegroundColor Red
        }
        
        $endTime = Get-Date
        $duration = [math]::Round(($endTime - $startTime).TotalSeconds, 2)
        
        # Combina tutto nel log finale
        $finalLog = @"
$logHeader
COMBINED OUTPUT:
$output

ERROR OUTPUT:
$errorOutput

===============================================
Completed: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Exit Code: $exitCode
Duration: $duration seconds
===============================================
"@
        
        $finalLog | Out-File -FilePath $LogFile -Encoding UTF8
        
        # Mostra output se richiesto
        if ($ShowOutput) {
            Write-Host "`n=== MATLAB Output ===" -ForegroundColor Yellow
            if (![string]::IsNullOrWhiteSpace($output)) {
                Write-Host $output
            }
            if (![string]::IsNullOrWhiteSpace($errorOutput)) {
                Write-Host "`n=== Errors ===" -ForegroundColor Red
                Write-Host $errorOutput
            }
        }
        
        # Determina successo
        $success = ($exitCode -eq 0) -and ([string]::IsNullOrWhiteSpace($errorOutput))
        
        # Se l'output contiene errori MATLAB specifici, considera fallimento
        if ($output -like "*Error*" -or $output -like "*error*") {
            $success = $false
        }
        
        $errorMessage = if (-not $success) {
            if (![string]::IsNullOrWhiteSpace($errorOutput)) { 
                $errorOutput 
            } elseif ($output -like "*Error*") {
                ($output -split "`n" | Where-Object { $_ -like "*Error*" }) -join "`n"
            } else { 
                "Command failed (Exit code: $exitCode)" 
            }
        } else { "" }
        
        # Risultato
        $result = @{
            Success = $success
            ExitCode = $exitCode
            ErrorMessage = $errorMessage
            LogFile = $LogFile
            Output = $output
            Duration = $duration
        }
        
        if ($success) {
            Write-Host "✓ MATLAB script completed successfully" -ForegroundColor Green
        } else {
            Write-Host "✗ MATLAB script failed" -ForegroundColor Red
            Write-Host "Error: $errorMessage" -ForegroundColor Red
        }
        
        return $result
    }
    finally {
        # Ripristina directory originale
        Set-Location $originalLocation
    }
}

# Funzione per processare un job MATLAB
function Execute-MatlabJobs($jobFolder) {
    Write-Host "Processing job: $jobFolder" -ForegroundColor Cyan
    
   
    $inputFolder = Join-Path $jobFolder "input"
    $paramsFile = Join-Path $inputFolder "params.json"
    $masksExt = ""
    $outputExt = ""
    $masksFolder = ""
    $originalImage = ""
    $bgColor = ""
    
    if (Test-Path $paramsFile) {
        try {
            $params = Get-Content $paramsFile | ConvertFrom-Json
            $masksExt = $params.mask_extension
            $outputExt = $params.output_extension
            $masksFolder = Join-Path $jobFolder $params.masks_dir
            $originalImage = Join-Path $jobFolder $params.original_image
            $bgColor = $params.bg_color
            $outputDir = Join-Path $jobFolder $params.output_steps_dirs.step1

            # Verifica che tutti i parametri necessari siano presenti
            $requiredParams = @('masksExt', 'outputExt', 'masksFolder', 'originalImage', 'bgColor', 'outputDir')
            $missingParams = @()

            foreach ($param in $requiredParams) {
              if (-not $params.PSObject.Properties.Name -contains $param -or [string]::IsNullOrEmpty($params.$param)) {
                $missingParams += $param
              }
            }

            if ($missingParams.Count -gt 0) {
              $errorMsg = "Missing required parameters in params.json: $($missingParams -join ', ')"
              Write-Error $errorMsg
              throw $errorMsg
            }
        }
        catch {
            Write-Warning "Could not parse params.json, using defaults"
        }
    }
    
    # Crea output folder
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Debug: mostra i parametri che verranno passati
    Write-Host "MATLAB Parameters:" -ForegroundColor Yellow
    Write-Host "  Masks folder: $masksFolder" -ForegroundColor Gray
    Write-Host "  Original image: $originalImage" -ForegroundColor Gray
    Write-Host "  Masks ext: $masksExt" -ForegroundColor Gray
    Write-Host "  Output ext: $outputExt" -ForegroundColor Gray
    Write-Host "  BG Color: $bgColor" -ForegroundColor Gray
    Write-Host "  Output dir: $outputDir" -ForegroundColor Gray
    
    # Esegui MATLAB script step 1
    $logFile = Join-Path $jobFolder "logs\matlab_execution_step1.log"
    
    $result = Invoke-MatlabScript -ScriptName "generate_new_example" `
        -Parameters @($masksFolder, $originalImage, $outputDir, $masksExt, $outputExt) `
        -LogFile $logFile `
        -TimeoutMinutes 30 `
        -AddPath `
        -ShowOutput
    
    $step1Success = $result.Success
    
    # Se step 1 è riuscito, esegui step 2 (export_alignments)
    if ($step1Success) {
        Write-Host "✓ Step 1 completed successfully, starting Step 2 (export_alignments)..." -ForegroundColor Green
        
        # Parametri per export_alignments
        $step1OutputDir = Join-Path $outputDir "m"
        $step2OutputDir = Join-Path $jobFolder $params.output_steps_dirs.step2
        $alignmentsFile = Join-Path $step2OutputDir "alignments.txt"
        $configFile = Join-Path $inputFolder "params.json"
        
        # Crea output folder per step 2
        if (-not (Test-Path $step2OutputDir)) {
            New-Item -ItemType Directory -Path $step2OutputDir -Force | Out-Null
        }
        
        Write-Host "Step 2 Parameters:" -ForegroundColor Yellow
        Write-Host "  Pics dir: $step1OutputDir" -ForegroundColor Gray
        Write-Host "  Output file: $alignmentsFile" -ForegroundColor Gray
        Write-Host "  Config file: $configFile" -ForegroundColor Gray
        
        $logFile2 = Join-Path $jobFolder "logs\matlab_execution_step2.log"
        
        $result2 = Invoke-MatlabScript -ScriptName "export_alignments" `
            -Parameters @($step1OutputDir, $alignmentsFile, $configFile) `
            -LogFile $logFile2 `
            -TimeoutMinutes 30 `
            -AddPath `
            -ShowOutput
        
        $overallSuccess = $step1Success -and $result2.Success
        
        if ($result2.Success) {
            Write-Host "✓ Step 2 completed successfully" -ForegroundColor Green
        } else {
            Write-Host "✗ Step 2 failed: $($result2.ErrorMessage)" -ForegroundColor Red
        }
    } else {
        Write-Host "✗ Step 1 failed, skipping Step 2" -ForegroundColor Red
        $overallSuccess = $false
    }
    
    # Salva risultato job completo
    $jobResult = @{
        JobFolder = $jobFolder
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Step1Success = $step1Success
        Step2Success = if ($step1Success) { $result2.Success } else { $false }
        OverallSuccess = $overallSuccess
        Step1Duration = $result.Duration
        Step2Duration = if ($step1Success) { $result2.Duration } else { 0 }
        Step1ErrorMessage = $result.ErrorMessage
        Step2ErrorMessage = if ($step1Success) { $result2.ErrorMessage } else { "Skipped due to Step 1 failure" }
        Step1LogFile = $result.LogFile
        Step2LogFile = if ($step1Success) { $result2.LogFile } else { "" }
    } | ConvertTo-Json -Depth 2
    
    $jobResult | Out-File -FilePath (Join-Path $jobFolder "job_result.json") -Encoding UTF8
    
    return $overallSuccess
}

# Salva il PID del processo corrente in un file per facilitare il controllo
$pidFile = Join-Path (Get-Location) "matlab_pipeline.pid"
$PID | Out-File -FilePath $pidFile -Encoding ASCII
Write-Host "Process ID: $PID (saved to $pidFile)"

# Se è specificato un TestJobPath, esegui direttamente su quella cartella e termina
if (-not [string]::IsNullOrEmpty($TestJobPath)) {
    Write-Host "Test mode: executing on job folder $TestJobPath" -ForegroundColor Magenta
    
    if (-not (Test-Path $TestJobPath)) {
        Write-Error "Test job path does not exist: $TestJobPath"
        exit 1
    }
    
    if (-not (IsJobReadyWindows $TestJobPath)) {
        Write-Error "Test job is not ready (missing required files)"
        exit 1
    }
    
    try {
        $success = Execute-MatlabJobs $TestJobPath
        if ($success) {
            Write-Host "✓ Test job completed successfully: $TestJobPath" -ForegroundColor Green
            exit 0
        } else {
            Write-Host "✗ Test job failed: $TestJobPath" -ForegroundColor Red
            exit 1
        }
    }
    catch {
        Write-Error "Exception processing test job $TestJobPath : $_"
        exit 1
    }
}

# Modalità normale: monitoraggio file system

# Determina il percorso da monitorare
if ([string]::IsNullOrEmpty($WatchPath)) {
    $watchPath = Join-Path (Get-Location) "jobs"
} else {
    $watchPath = $WatchPath
}

Write-Host "Target watch path: $watchPath"

# Controlla se il percorso è valido
try {
    $absolutePath = Resolve-Path $watchPath -ErrorAction SilentlyContinue
    if (-not $absolutePath) {
        # Il percorso non esiste, prova a crearlo
        Write-Host "Path does not exist. Attempting to create: $watchPath"
        $null = New-Item -ItemType Directory -Path $watchPath -Force
        $absolutePath = Resolve-Path $watchPath
    }
    
    # Crea il FileSystemWatcher
    $watcher = New-Object System.IO.FileSystemWatcher
    $watcher.Path = $absolutePath
    Write-Host "Successfully monitoring: $($watcher.Path)"
    Write-Host "Press Ctrl+C to stop monitoring..."
}
catch {
    Write-Error "Failed to set up file system watcher: $_"
    Write-Host "Please ensure the path exists or you have permissions to create it"
    exit 1
}

function IsJobReadyWindows($jobFolder) {
    $inputFolder = Join-Path $jobFolder "input\masks"
    $origImg = Join-Path $jobFolder "input\original.jpg"
    $paramsFile = Join-Path $jobFolder "input\params.json"

    $ready = (Test-Path $inputFolder) -and (Test-Path $origImg) -and (Test-Path $paramsFile)
    if (-not $ready) {
        Write-Host "  Missing files:" -ForegroundColor Gray
        if (-not (Test-Path $inputFolder)) { Write-Host "    - masks" -ForegroundColor Gray }
        if (-not (Test-Path $origImg)) { Write-Host "    - original.jpg" -ForegroundColor Gray }
        if (-not (Test-Path $paramsFile)) { Write-Host "    - params.json" -ForegroundColor Gray }
    }
    
    return $ready
}

# Lista globale dei job in attesa (semplice array)
$Global:PendingJobsList = [System.Collections.ArrayList]::new()

# Registra l'evento - usa variabile globale
$job = Register-ObjectEvent $watcher Created -Action {
    $jobFolder = $Event.SourceEventArgs.FullPath
    Write-Host "New job detected: $jobFolder" -ForegroundColor Yellow
    
    # Aggiungi job alla lista globale
    $jobInfo = @{
        Path = $jobFolder
        Timestamp = Get-Date
    }
    
    $null = $Global:PendingJobsList.Add($jobInfo)
    Write-Host "Added to pending jobs queue: $jobFolder" -ForegroundColor Gray
}

# Gestione pulita dell'interruzione
try {
    Write-Host "Monitoring started. Press Ctrl+C to stop."
    
    # Loop principale che controlla i job pendenti
    while ($true) {
        Start-Sleep -Seconds 2
        
        # Controlla se esiste un file di stop
        $stopFile = Join-Path (Get-Location) "stop_matlab_pipeline"
        if (Test-Path $stopFile) {
            Write-Host "Stop file detected. Shutting down..."
            Remove-Item $stopFile -Force
            break
        }
        
        # Processa i job pendenti dalla lista globale
        $jobsToRemove = [System.Collections.ArrayList]::new()
        
        for ($i = 0; $i -lt $Global:PendingJobsList.Count; $i++) {
            $jobInfo = $Global:PendingJobsList[$i]
            if ($null -eq $jobInfo) { continue }
            
            $jobFolder = $jobInfo.Path
            $jobAge = ((Get-Date) - $jobInfo.Timestamp).TotalSeconds
            
            # Timeout per job vecchi (evita accumulo)
            if ($jobAge -gt 300) {  # 5 minuti
                Write-Warning "Job timeout: $jobFolder (waited $jobAge seconds)"
                $null = $jobsToRemove.Add($i)
                continue
            }
            
            # Controlla se il job è pronto
            if (IsJobReadyWindows $jobFolder) {
                Write-Host "Job is ready, processing: $jobFolder" -ForegroundColor Green
                
                try {
                    $success = Execute-MatlabJobs $jobFolder
                    if ($success) {
                        Write-Host "✓ Job completed successfully: $jobFolder" -ForegroundColor Green
                    } else {
                        Write-Host "✗ Job failed: $jobFolder" -ForegroundColor Red
                    }
                }
                catch {
                    Write-Error "Exception processing job $jobFolder : $_"
                }

                $null = $jobsToRemove.Add($i)
            }
            else {
                Write-Host "Job not ready yet: $jobFolder (waiting $([math]::Round([double]$jobAge, 1))s)" -ForegroundColor Gray
            }
        }
        
        # Rimuovi job processati o scaduti (in ordine inverso per mantenere indici)
        $jobsToRemove.Sort()
        for ($i = $jobsToRemove.Count - 1; $i -ge 0; $i--) {
            $Global:PendingJobsList.RemoveAt($jobsToRemove[$i])
        }
        
        if ($Global:PendingJobsList.Count -gt 0) {
            Write-Host "Pending jobs: $($Global:PendingJobsList.Count)" -ForegroundColor Cyan
        }
    }
} finally {
    # Cleanup quando lo script viene interrotto
    Write-Host "`nCleaning up..."
    
    if ($job) {
        Unregister-Event -SourceIdentifier $job.Name -ErrorAction SilentlyContinue
        Remove-Job -Job $job -ErrorAction SilentlyContinue
    }
    
    if ($watcher) {
        $watcher.Dispose()
    }
    
    # Rimuovi il file PID
    if (Test-Path $pidFile) {
        Remove-Item $pidFile -Force
    }
    
    Write-Host "Monitoring stopped."
}