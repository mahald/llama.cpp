#!/bin/bash
# Run a benchmark on a VPS instance
# Usage: ./bench-vps.sh <ssh-host> <ssh-port> <mode> <ctx> [codebook] [extra-args]
#
# Modes:
#   ppl     - perplexity measurement
#   kld     - KL divergence measurement
#   speed   - llama-bench decode speed
#
# Examples:
#   ./bench-vps.sh root@207.174.105.41 21075 ppl 2048
#   ./bench-vps.sh root@207.174.105.41 21075 kld 2048 /root/codebooks/cb_iter050.bin
#   ./bench-vps.sh root@207.174.105.41 21075 speed 2048

set -e

SSH_HOST="${1:?Usage: $0 <host> <port> <ppl|kld|speed> <ctx> [codebook] [extra-args]}"
SSH_PORT="${2:?}"
MODE="${3:?}"
CTX="${4:?}"
CODEBOOK="${5:-}"
EXTRA="${6:-}"

SSH="ssh -p $SSH_PORT $SSH_HOST"
BIN="/root/llama-turbo/build/bin"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
WIKI="/root/wikitext-2-raw/wiki.test.raw"

# compute chunks: use enough to cover wikitext
TOKENS=287646  # wikitext-2 token count for this model (approx)
CHUNKS=$(( TOKENS / CTX ))
[ $CHUNKS -lt 1 ] && CHUNKS=1
[ $CHUNKS -gt 64 ] && CHUNKS=64

# build env vars for codebook override
ENV=""
if [ -n "$CODEBOOK" ]; then
	ENV="TURBO_TCQ_CB=$CODEBOOK TURBO_TCQ_ALPHA_V=1.04"
fi

case "$MODE" in
	ppl)
		echo "=== PPL: ctx=$CTX chunks=$CHUNKS ==="
		$SSH "$ENV $BIN/llama-perplexity \
			-m $MODEL -ctk turbo3_tcq -ctv turbo3_tcq \
			-f $WIKI -c $CTX --chunks $CHUNKS -ngl 99 -t 1 $EXTRA 2>&1" \
			| grep -E '(Final estimate|perplexity)'
		;;
	kld)
		echo "=== KLD: ctx=$CTX chunks=$CHUNKS ==="
		$SSH "$ENV $BIN/llama-perplexity --kl-divergence \
			-m $MODEL -ctk turbo3_tcq -ctv turbo3_tcq \
			-f $WIKI -c $CTX --chunks $CHUNKS -ngl 99 -t 1 $EXTRA 2>&1" \
			| grep -E '(KL|divergence|Final)'
		;;
	speed)
		echo "=== Speed: ctx=$CTX ==="
		$SSH "$BIN/llama-bench \
			-m $MODEL -ctk turbo3_tcq -ctv turbo3_tcq \
			-c $CTX -ngl 99 -t 1 $EXTRA 2>&1" \
			| grep -E '(tg64|model)'
		;;
	*)
		echo "Unknown mode: $MODE (use ppl, kld, or speed)"
		exit 1
		;;
esac
