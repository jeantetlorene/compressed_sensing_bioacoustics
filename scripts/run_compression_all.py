import subprocess

jobs = {
    "mp3":  ["32k", "56k", "96k"],
    "opus": ["8k", "48k", "112k"],
    "aac":  ["8k", "40k", "96k"],
    "ogg":  ["0", "4", "8"],
    "flac": ["0", "2", "8"],
}

for method, params in jobs.items():
    for param in params:
        print(f"=== {method} @ {param} ===")
        subprocess.run([
            "python",
            "scripts/run_compression.py",
            "--species", "thyolo",
            "--method-compression", method,
            "--parameter-compression", param,
        ], check=True)