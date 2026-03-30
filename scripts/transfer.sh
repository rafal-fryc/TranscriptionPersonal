#!/usr/bin/env bash
set -euo pipefail

# Transfer a recording to the Jetson for transcription.
# Configure via environment variables or defaults.
#
# Usage:
#   ./scripts/transfer.sh recording_20260328_143000.wav
#   JETSON_HOST=100.x.x.x ./scripts/transfer.sh recording.wav
#
# With Tailscale, JETSON_HOST is typically the Tailscale hostname or IP.

JETSON_HOST="${JETSON_HOST:-jetson}"
JETSON_USER="${JETSON_USER:-user}"
JETSON_DIR="${JETSON_DIR:-~/recordings/inbox}"

if [ $# -eq 0 ]; then
    echo "Usage: transfer.sh <recording.wav> [recording2.wav ...]"
    echo ""
    echo "Environment variables:"
    echo "  JETSON_HOST  Hostname or IP (default: jetson)"
    echo "  JETSON_USER  SSH user (default: user)"
    echo "  JETSON_DIR   Remote directory (default: ~/recordings/inbox)"
    exit 1
fi

# Ensure remote directory exists
ssh "${JETSON_USER}@${JETSON_HOST}" "mkdir -p ${JETSON_DIR}"

for FILE in "$@"; do
    if [ ! -f "$FILE" ]; then
        echo "Error: $FILE not found"
        exit 1
    fi
    BASENAME=$(basename "$FILE")
    SIZE=$(du -h "$FILE" | cut -f1)
    echo "Transferring ${BASENAME} (${SIZE}) to ${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/"
    scp "$FILE" "${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/${BASENAME}"
done

echo "Done. Run on Jetson:"
echo "  transcribe ${JETSON_DIR}/<filename>.wav"
