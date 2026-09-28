$jobs = [ordered]@{
    "mp3"  = @("32k", "64k", "128k")
    "opus" = @("8k", "56k", "160k")
    "aac"  = @("8k", "56k", "144k")
    "ogg"  = @("0", "4", "9")
    "flac" = @("0", "2", "4")
}

$jobs = [ordered]@{

    "aac"  = @( "144k")
    "ogg"  = @("0", "4", "9")
    "flac" = @("0", "2", "4")
}

foreach ($method in $jobs.Keys) {
    foreach ($param in $jobs[$method]) {

        Write-Host "=== $method @ $param ==="

        python scripts/run_compression.py `
            --species ptw `
            --method-compression $method `
            --parameter-compression $param
    }
}