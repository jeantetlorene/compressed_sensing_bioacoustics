"""
clean_cs_folder.py
------------------
Deletes files in `cs_0.1` whose reconstructed counterpart already exists
in `cs_reconstructed_0.1`.
 
Mapping:
  cs_reconstructed_0.1/<base>_reconstructed
  cs_0.1/<base>_compressed          ← this one gets deleted
 
Usage:
  python clean_cs_folder.py                        # dry-run (safe, no deletion)
  python clean_cs_folder.py --confirm              # actually delete files
  python clean_cs_folder.py --src /path/to/cs_0.1 --ref /path/to/cs_reconstructed_0.1 --confirm
"""
 
import argparse
import os
from pathlib import Path
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Clean cs_0.1 of already-reconstructed files.")
    parser.add_argument(
        "--src",
        default="Data/Bats/Compressed_Audio/cs_0.1",
        help="Folder containing the *_compressed source files (default: cs_0.1)",
    )
    parser.add_argument(
        "--ref",
        default="Data/Bats/Compressed_Audio/cs_reconstructed_0.1",
        help="Folder containing the *_reconstructed files (default: cs_reconstructed_0.1)",
    )
    parser.add_argument(
        "--reconstructed-suffix",
        default="_reconstructed",
        help="Suffix that marks a reconstructed file (default: _reconstructed)",
    )
    parser.add_argument(
        "--compressed-suffix",
        default="_compressed",
        help="Suffix that marks a compressed source file (default: _compressed)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete files. Without this flag the script runs as a dry-run.",
    )
    return parser.parse_args()
 
 
def get_base_name(filename: str, suffix: str) -> str | None:
    """Strip the given suffix (and optional file extension) from a filename."""
    stem = Path(filename).stem          # drop file extension if any
    name = Path(filename).name          # keep extension logic separate
 
    # Try stem first (handles files with extensions)
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
 
    # Try full name (handles extension-less files)
    if name.endswith(suffix):
        return name[: -len(suffix)]
 
    return None
 
 
def main():
    args = parse_args()
 
    src_dir = Path(args.src)
    ref_dir = Path(args.ref)
 
    if not src_dir.is_dir():
        print(f"[ERROR] Source folder not found: {src_dir.resolve()}")
        return
    if not ref_dir.is_dir():
        print(f"[ERROR] Reference folder not found: {ref_dir.resolve()}")
        return
 
    mode = "DRY-RUN" if not args.confirm else "DELETE"
    print(f"Mode           : {mode}")
    print(f"Source folder  : {src_dir.resolve()}")
    print(f"Reference folder: {ref_dir.resolve()}")
    print(f"Compressed suffix  : {args.compressed_suffix}")
    print(f"Reconstructed suffix: {args.reconstructed_suffix}")
    print("-" * 60)
 
    # 1. Collect base IDs from the reconstructed folder
    reconstructed_bases: set[str] = set()
    for entry in ref_dir.iterdir():
        if not entry.is_file():
            continue
        base = get_base_name(entry.name, args.reconstructed_suffix)
        if base is not None:
            reconstructed_bases.add(base)
 
    print(f"Reconstructed files found : {len(reconstructed_bases)}")
 
    # 2. Walk the source folder and find matching compressed files
    to_delete: list[Path] = []
    unmatched_src: list[Path] = []
 
    for entry in src_dir.iterdir():
        if not entry.is_file():
            continue
        base = get_base_name(entry.name, args.compressed_suffix)
        if base is None:
            continue  # not a compressed file — skip
        if base in reconstructed_bases:
            to_delete.append(entry)
        else:
            unmatched_src.append(entry)
 
    print(f"Compressed files to delete: {len(to_delete)}")
    print(f"Compressed files to keep  : {len(unmatched_src)}")
    print("-" * 60)
 
    # 3. Delete (or report)
    deleted = 0
    errors = 0
    for path in sorted(to_delete):
        if args.confirm:
            try:
                path.unlink()
                print(f"  DELETED  {path.name}")
                deleted += 1
            except OSError as e:
                print(f"  ERROR    {path.name} — {e}")
                errors += 1
        else:
            print(f"  [dry-run] would delete  {path.name}")
 
    print("-" * 60)
    if args.confirm:
        print(f"Done. Deleted: {deleted}  |  Errors: {errors}")
    else:
        print(f"Dry-run complete. {len(to_delete)} file(s) would be deleted.")
        print("Re-run with --confirm to actually delete them.")
 
 
if __name__ == "__main__":
    main()
