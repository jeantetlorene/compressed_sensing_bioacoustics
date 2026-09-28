$species = "ptw"
$speciesFolder = "C:/Users/loren/Documents/Postdoc/Compressed_sensing/Data/Ptw"

$jobs = [ordered]@{
    "mp3"  = @("32k", "64k", "128k")
    "opus" = @("8k", "56k", "160k")
    "encodec" = @("6.0", "12.0", "24.0")
    "aac"  = @("8k","56k", "144k")
    "ogg"  = @("0", "4", "9")
    "flac" = @("0", "2", "4")
}

$jobs = [ordered]@{
    "encodec" = @("6.0", "12.0", "24.0")
    "aac"  = @("8k","56k", "144k")
    "ogg"  = @("0", "4", "9")
    "flac" = @("0", "2", "4")
}

# Mirrors the audio_extension logic in scripts/train_cnn.py so the cached
# amplitudes folder can be located and removed after each combo.
$extensionByMethod = @{
    "cs"      = ".npy"
    "mp3"     = ".mp3"
    "aac"     = ".aac"
    "opus"    = ".opus"
    "ogg"     = ".ogg"
    "flac"    = ".flac"
    "encodec" = ".wav"
}

foreach ($method in $jobs.Keys) {
    foreach ($param in $jobs[$method]) {

        Write-Host "=== $method @ $param ==="

        python scripts/train_cnn.py `
            --species $species `
            --method-compression $method `
            --parameter-compression $param

        $extension = $extensionByMethod[$method]
        $amplitudesFolder = Join-Path $speciesFolder "amplitudes_to_predict_$extension"

        if (Test-Path $amplitudesFolder) {
            Write-Host "Deleting cached amplitudes folder: $amplitudesFolder"
            Remove-Item -Recurse -Force $amplitudesFolder
        }
    }
}
