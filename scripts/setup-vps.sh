#!/bin/bash
# Setup script for deploying TurboQuant to a fresh VPS (vast.ai RTX 3090)
# Usage: ./setup-vps.sh <ssh-host> <ssh-port> [dorei-host]
#
# Example: ./setup-vps.sh root@207.174.105.41 21075 root@dorei
#
# Prerequisites:
#   - SSH key auth to both VPS and dorei (or model source)
#   - Local git repo on master branch
#   - Model file on dorei at /root/Qwen3.5-27B-heretic.Q6_K.gguf

set -e

SSH_HOST="${1:?Usage: $0 <ssh-host> <ssh-port> [dorei-host]}"
SSH_PORT="${2:?Usage: $0 <ssh-host> <ssh-port> [dorei-host]}"
DOREI="${3:-root@dorei}"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
WIKITEXT="/root/wikitext-2-raw/wiki.test.raw"
REMOTE_DIR="/root/llama-turbo"

SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new $SSH_HOST"
SCP="scp -P $SSH_PORT"

echo "=== TurboQuant VPS Setup ==="
echo "Target: $SSH_HOST:$SSH_PORT"
echo ""

# Step 1: Silence SSH banner and motd
echo "[1/6] Clearing SSH banner/motd..."
$SSH "echo -n > /etc/banner 2>/dev/null; chmod -x /etc/update-motd.d/* 2>/dev/null; echo -n > /etc/motd 2>/dev/null; echo done"

# Step 2: Archive and upload code
echo "[2/6] Uploading code..."
TMPARCHIVE=$(mktemp /tmp/turbo-XXXXXX.tar.gz)
git archive master | gzip > "$TMPARCHIVE"
$SCP "$TMPARCHIVE" "$SSH_HOST:/root/turbo-master.tar.gz"
rm "$TMPARCHIVE"
$SSH "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && tar xzf /root/turbo-master.tar.gz && rm /root/turbo-master.tar.gz"

# Step 3: Build
echo "[3/6] Building (this takes a few minutes)..."
$SSH "cd $REMOTE_DIR && cmake -B build \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_COMPILER=\$(which nvcc) \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3 && \
  cmake --build build -j\$(nproc) 2>&1 | tail -3"

# Step 4: Verify build
echo "[4/6] Verifying build..."
$SSH "$REMOTE_DIR/build/bin/llama-perplexity --help 2>&1 | head -2"

# Step 5: Upload wikitext (small, relay through local)
echo "[5/6] Uploading wikitext..."
TMPWIKI=$(mktemp /tmp/wiki-XXXXXX.raw)
scp "$DOREI:$WIKITEXT" "$TMPWIKI"
$SSH "mkdir -p /root/wikitext-2-raw"
$SCP "$TMPWIKI" "$SSH_HOST:$WIKITEXT"
rm "$TMPWIKI"
CHECKSUM=$($SSH "md5sum $WIKITEXT | cut -d' ' -f1")
if [ "$CHECKSUM" = "7c0137fc034ddbc56a296bce31b4f7fb" ]; then
  echo "  Wikitext checksum OK"
else
  echo "  WARNING: Wikitext checksum mismatch: $CHECKSUM"
fi

# Step 6: Transfer model (direct dorei→VPS, 21GB)
echo "[6/6] Transferring model from $DOREI (21GB, this takes a while)..."
MODEL_SIZE=$(ssh "$DOREI" "stat -c%s $MODEL")
echo "  Source size: $MODEL_SIZE bytes"

# Use rsync with partial for resume support, fall back to scp
ssh "$DOREI" "scp -P $SSH_PORT $MODEL $SSH_HOST:$MODEL"

# Verify transfer
VPS_SIZE=$($SSH "stat -c%s $MODEL 2>/dev/null || echo 0")
if [ "$VPS_SIZE" = "$MODEL_SIZE" ]; then
  echo "  Model transfer OK ($VPS_SIZE bytes)"
else
  echo "  ERROR: Model truncated ($VPS_SIZE / $MODEL_SIZE bytes)"
  echo "  Re-run: ssh $DOREI 'scp -P $SSH_PORT $MODEL $SSH_HOST:$MODEL'"
  exit 1
fi

echo ""
echo "=== Setup complete ==="
echo "SSH:   ssh -p $SSH_PORT $SSH_HOST"
echo "Build: $REMOTE_DIR/build/bin/"
echo "Model: $MODEL (transferring...)"
echo "Wiki:  $WIKITEXT"
echo ""
echo "Quick benchmark command (after model transfer):"
echo "  ssh -p $SSH_PORT $SSH_HOST '$REMOTE_DIR/build/bin/llama-perplexity \\"
echo "    -m $MODEL -ctk turbo3_tcq -ctv turbo3_tcq \\"
echo "    -f $WIKITEXT -c 2048 --chunks 8 -ngl 99 -t 1'"

wait $MODEL_PID 2>/dev/null && echo "Model transfer complete!" || true
