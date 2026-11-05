# Navi

I built Navi for my grandpa, whose eyesight keeps getting worse. He wears Ray-Ban Meta glasses, a server on a Mac does the seeing, and a small haptic belt around his waist buzzes on the side where something is in his way. He can also ask out loud where he left his keys, and Navi answers with a place and a direction.

![architecture](docs/architecture.svg)

## How it works

The iOS app in `ios/` streams camera and microphone from the glasses to the server over Meta's Wearables Device Access Toolkit.

On the server (`metasight/`), a fast safety loop splits the view into sectors and gives each one a distance. A slower memory loop builds a metric 3D map of the home and pins the objects it sees, which is how "where are my keys?" gets its answer.

The belt runs ESP32 firmware from `firmware/belt/` and pulses faster as an obstacle gets closer. The web client in `web/` (TypeScript and three.js) shows the live map, the sectors and the belt.

## Meta models it runs on

| Job | Model |
| --- | --- |
| 3D map from keyframes | MapAnything |
| Image features for matching places and objects | DINOv2 |
| One-sentence answers to questions | Muse Spark |
| Reviewing what changed in the map | Muse Wonder |
| Speech to text | Muse Voice Transcribe |
| Finding small objects | SAM 3.1 |
| Local language model, runs offline | Muse Glimmer |
| Cleaning up microphone audio | Meta denoiser (dns48) |

Each model sits behind a small interface and is picked in `metasight.toml`.

## Run it

You need a Mac with Apple silicon, Python 3.12 with `uv`, and Node with `pnpm`.

```sh
uv venv --python 3.12 && uv pip install -e ".[dev]"
(cd web && pnpm install && pnpm build)
scripts/run.sh                   # http://127.0.0.1:8444, open it in a browser
```

A phone needs HTTPS to use the camera. Make a self-signed certificate for your LAN address with `scripts/make_cert.sh`, run `scripts/run.sh CERT KEY`, then open `https://<mac ip>:8443` on the phone and accept the certificate.

Without a camera, replay a video: `python scripts/replay.py clip.mp4 --ask "where is the remote"`. To build a map offline: `python scripts/build_map.py clip.mp4`.

The Meta models need `META_API_KEY` in the environment or the macOS Keychain. Keys are never written to a file. Model weights, room scans and recorded footage are not in this repo.

Tests: `python -m pytest`.

## Simulation numbers

The belt patterns were tuned in a walking simulation (results in `bench/results/`). These come from a simulated user model, not a human study.

## License

MIT, see `LICENSE`. The iOS app builds on Meta's Camera Access sample; see `ios/THIRD_PARTY.md` for its terms.
