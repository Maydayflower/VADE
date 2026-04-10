import pandas as pd
import os
import subprocess

DATA_DIR = "/workspace/compare_model/bart/clip/data"
VAL_TSV = os.path.join(DATA_DIR, "val_pairs_emotion.tsv")
TRAIN_TSV = os.path.join(DATA_DIR, "train_pairs_emotion.tsv")
QWEN_CAPTION_SCRIPT = "/workspace/compare_model/bart/clip/tools/qwen_caption.py"

def find_error_rows(input_tsv):
    """Return DataFrame of rows where any column contains '[ERROR_IN_QWEN3_API]'."""
    df = pd.read_csv(input_tsv, sep="\t", encoding="utf-8", keep_default_na=False)
    error_mask = df.apply(lambda row: row.astype(str).str.contains(r'\[ERROR_IN_QWEN3_API\]').any(), axis=1)
    error_rows = df[error_mask]
    return error_rows

def save_errors(error_rows, error_tsv):
    if not error_rows.empty:
        error_rows.to_csv(error_tsv, sep='\t', index=False, encoding='utf-8')
        return True
    return False

def process_error_file(error_tsv, backend="qwen2.5"):
    TMP_SRC = os.path.join(DATA_DIR, "__src_qwen_tmp.tsv")
    TMP_OUT = error_tsv.replace(".tsv", "_repaired.tsv")
    # Copy error tsv to temporary source
    os.system(f"cp {error_tsv} {TMP_SRC}")
    # Backup original VAL_TSV used by qwen_caption.py
    bak_val = VAL_TSV + '.bak'
    os.system(f"mv {VAL_TSV} {bak_val}")
    os.system(f"cp {TMP_SRC} {VAL_TSV}")
    # Remove old output if exists
    OUT_TSV = os.path.join(DATA_DIR, "train_pairs_emotion.tsv")
    if os.path.exists(OUT_TSV):
        os.remove(OUT_TSV)
    # Run the script (output will be written to train_pairs_emotion.tsv)
    subprocess.run([
        "python3", QWEN_CAPTION_SCRIPT, "--backend", backend
    ])
    # Move the output to the replacement file
    if os.path.exists(OUT_TSV):
        os.rename(OUT_TSV, TMP_OUT)
    # Restore original VAL_TSV
    os.system(f"mv {bak_val} {VAL_TSV}")
    if os.path.exists(TMP_OUT):
        return TMP_OUT
    return None

def main():
    # 处理所有行中存在[ERROR_IN_QWEN3_API]的数据
    for src_tsv_basename in ["val_pairs_emotion.tsv", "train_pairs_emotion.tsv"]:
        src_tsv = os.path.join(DATA_DIR, src_tsv_basename)
        error_tsv = src_tsv.replace(".tsv", "_error.tsv")
        error_rows = find_error_rows(src_tsv)
        has_error = save_errors(error_rows, error_tsv)
        if has_error:
            print(f"Saved error rows from {src_tsv_basename} to {os.path.basename(error_tsv)}")
            repaired_path = process_error_file(error_tsv, "qwen2.5")
            if repaired_path:
                print(f"Processed error samples for {src_tsv_basename}, results saved in {os.path.basename(repaired_path)}")
            else:
                print(f"Failed to repair error samples for {src_tsv_basename}")
        else:
            print(f"No '[ERROR_IN_QWEN3_API]' rows found in {src_tsv_basename}")

if __name__ == "__main__":
    main()
