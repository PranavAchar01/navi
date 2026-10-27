#!/bin/sh
# Self-signed certificate for the LAN, so a phone browser will allow the camera (HTTPS only).
# Usage: scripts/make_cert.sh [LAN_IP]   (default: this Mac's wifi address)
set -e
cd "$(dirname "$0")/.."
IP="${1:-$(ipconfig getifaddr en0 || ipconfig getifaddr en1)}"
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 30 -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=metasight" -addext "subjectAltName=IP:$IP,DNS:localhost,IP:127.0.0.1"
echo "certificate for $IP written to certs/ (gitignored)"
echo "run: scripts/run.sh certs/cert.pem certs/key.pem"
