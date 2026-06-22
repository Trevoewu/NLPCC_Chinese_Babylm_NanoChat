#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${TARGET_ROOT:-/mnt/proj}"
BABYLLM_DIR="${BABYLLM_DIR:-$TARGET_ROOT/babyllm}"
EVAL_DIR="${EVAL_DIR:-$TARGET_ROOT/chinese-babylm-eval-pipeline}"

NANOCHAT_REPO_URL="${NANOCHAT_REPO_URL:-https://github.com/karpathy/nanochat.git}"
EVAL_REPO_URL="${EVAL_REPO_URL:-https://github.com/SiyuanSong2004/chinese-babylm-eval-pipeline.git}"

OVERLAY_BABYLLM="$ROOT_DIR/overlays/babyllm"
OVERLAY_EVAL="$ROOT_DIR/overlays/chinese-babylm-eval-pipeline"

log() {
    printf '[restore] %s\n' "$*"
}

ensure_git_repo() {
    local url="$1"
    local dir="$2"
    local label="$3"

    if [ -d "$dir/.git" ]; then
        log "$label exists: $dir"
        return
    fi

    if [ -e "$dir" ] && [ "$(find "$dir" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)" -gt 0 ]; then
        if [ "${RESTORE_IN_PLACE:-0}" = "1" ]; then
            log "$label target is non-empty; RESTORE_IN_PLACE=1, keeping it and applying overlay: $dir"
            return
        fi
        local fallback="${dir}_recovered"
        if [ -e "$fallback" ]; then
            fallback="${dir}_recovered_$(date +%Y%m%d_%H%M%S)"
        fi
        log "$label target is non-empty and not a git repo: $dir"
        log "using non-destructive fallback target: $fallback"
        if [ "$label" = "babyllm" ]; then
            BABYLLM_DIR="$fallback"
        else
            EVAL_DIR="$fallback"
        fi
        dir="$fallback"
    fi

    mkdir -p "$(dirname "$dir")"
    log "cloning $label from $url to $dir"
    git clone "$url" "$dir"
}

copy_overlay() {
    local src="$1"
    local dst="$2"
    if [ ! -d "$src" ]; then
        log "overlay missing, skipping: $src"
        return
    fi
    mkdir -p "$dst"
    log "applying overlay: $src -> $dst"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$src"/ "$dst"/
    else
        cp -a "$src"/. "$dst"/
    fi
}

link_dataset_if_found() {
    local target="$1"
    local explicit="${CHINESE_BABYLM_DATA_DIR:-}"
    local found=""

    if [ -n "$explicit" ] && [ -e "$explicit" ]; then
        found="$explicit"
    else
        for candidate in \
            "$TARGET_ROOT/chinese-babylm/data/raw/huggingface/chinese-babylm-org/babylm-zho-100M" \
            "$TARGET_ROOT/chinese-babylm/data/processed/tokenizer_corpus/babylm_zho_100m.txt" \
            "$TARGET_ROOT/chinese-babylm/data/processed/pretrain_splits/babylm_zho_100m_train_excluding_fixed_eval.txt"
        do
            if [ -e "$candidate" ]; then
                found="$candidate"
                break
            fi
        done

        if [ -z "$found" ]; then
            found="$(find "$TARGET_ROOT" -maxdepth 7 \( \
                -name 'dataset_info.json' -o \
                -name 'state.json' -o \
                -name 'train-*.parquet' -o \
                -name '*babylm*zho*100m*.txt' \
            \) 2>/dev/null | head -n 1 || true)"
            if [ -n "$found" ] && [ -f "$found" ]; then
                found="$(dirname "$found")"
            fi
        fi
    fi

    if [ -z "$found" ]; then
        log "Chinese BabyLM dataset not found under $TARGET_ROOT; set CHINESE_BABYLM_DATA_DIR before training."
        return
    fi

    mkdir -p "$target/data"
    if [ ! -e "$target/data/babylm-zho-100M" ]; then
        ln -s "$found" "$target/data/babylm-zho-100M"
        log "linked dataset: $target/data/babylm-zho-100M -> $found"
    else
        log "dataset link/path already exists: $target/data/babylm-zho-100M"
    fi
}

write_notes() {
    local notes="$BABYLLM_DIR/RECOVERY_NOTES.md"
    cat > "$notes" <<EOF
# BabyLLM Recovery Notes

Recovered on: $(date -Is)

Code target:

\`\`\`text
$BABYLLM_DIR
\`\`\`

Eval pipeline target:

\`\`\`text
$EVAL_DIR
\`\`\`

Main restored files:

- \`runs/chinese_babylm.sh\`
- \`scripts/prepare_chinese_babylm.py\`
- \`scripts/base_train.py\`
- \`nanochat/tokenizer.py\`
- \`nanochat/common.py\`
- \`nanochat/loss_eval.py\`
- \`eval_configs/*.yaml\`
- \`docs/*.md\`
- Chinese BabyLM eval pipeline nanochat adapter files

Recommended setup:

\`\`\`bash
cd $BABYLLM_DIR
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install datasets pyarrow transformers accelerate scikit-learn scipy pandas
\`\`\`

Training smoke command:

\`\`\`bash
cd $BABYLLM_DIR
export HF_ENDPOINT=https://hf-mirror.com
export NANOCHAT_BASE_DIR=$BABYLLM_DIR/chinese_cache
RUN_BASE_TRAIN=0 bash runs/chinese_babylm.sh
\`\`\`
EOF
    log "wrote $notes"
}

main() {
    mkdir -p "$TARGET_ROOT"

    ensure_git_repo "$NANOCHAT_REPO_URL" "$BABYLLM_DIR" "babyllm"
    copy_overlay "$OVERLAY_BABYLLM" "$BABYLLM_DIR"
    link_dataset_if_found "$BABYLLM_DIR"

    ensure_git_repo "$EVAL_REPO_URL" "$EVAL_DIR" "eval"
    copy_overlay "$OVERLAY_EVAL" "$EVAL_DIR"

    if [ -f "$ROOT_DIR/evaluation_data.tgz" ] && [ ! -d "$EVAL_DIR/evaluation_data" ]; then
        log "extracting evaluation_data.tgz into $EVAL_DIR"
        tar -xzf "$ROOT_DIR/evaluation_data.tgz" -C "$EVAL_DIR"
    fi

    mkdir -p "$BABYLLM_DIR/eval_configs" "$BABYLLM_DIR/runs/logs"
    if [ -d "$BABYLLM_DIR/eval_configs" ]; then
        cp "$BABYLLM_DIR"/eval_configs/*.yaml "$EVAL_DIR"/ 2>/dev/null || true
    fi

    write_notes

    log "done"
    log "babyllm: $BABYLLM_DIR"
    log "eval pipeline: $EVAL_DIR"
}

main "$@"
