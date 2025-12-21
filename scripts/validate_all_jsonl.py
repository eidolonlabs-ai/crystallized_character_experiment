import json
import os
import sys
def validate_jsonl_file(filepath):
    errors = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Flag comment lines or lines not starting with valid JSON
            if line.startswith('//') or not (line.startswith('{') or line.startswith('[')):
                errors.append(f"{filepath}: Line {i}: Invalid format or comment line detected")
                continue
            try:
                json.loads(line)
            except Exception as e:
                errors.append(f"{filepath}: Line {i}: {e}")
    return errors

def main():
    # List of all JSONL files to check
    files = [
        "raw_data/training_data_baseline.jsonl",
        "raw_data/training_data_baseline_augmented.jsonl",
        "raw_data/training_data_baseline_full.jsonl"
    ]
    # Add all train/valid.jsonl files in subfolders
    for root, dirs, files_in_dir in os.walk("raw_data"):
        for fname in files_in_dir:
            if fname.endswith(".jsonl") and fname not in files:
                files.append(os.path.join(root, fname))
    all_errors = []
    for f in files:
        if os.path.exists(f):
            errs = validate_jsonl_file(f)
            all_errors.extend(errs)
        else:
            all_errors.append(f"{f}: File not found")
    if all_errors:
        print("JSONL validation errors found:")
        for err in all_errors:
            print(err)
        sys.exit(1)
    else:
        print("All JSONL files are valid.")

if __name__ == "__main__":
    main()
