# Third-party code in the iOS app

## Depth Anything V2 Metric-Indoor Small (Apache-2.0)

- Source: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf, the
  same weights the server's safety loop runs (metasight/safety.py `DEPTH_REPO`).
- Licence: Apache-2.0 (the Small model only; Base and Large are CC-BY-NC-4.0 and not used).
- What we ship: `CameraAccess/MetaSight/Models/ReflexDepth.mlpackage`, converted by
  `scripts/coreml_depth.py` (fixed 294 x 518 input, ImageNet normalisation in the graph, int8
  weights). The model's weights are unchanged apart from that quantisation; no code is vendored.

## Microsoft Soundscape (MIT)

- Source: https://github.com/microsoft/soundscape (archived; community fork:
  https://github.com/soundscape-community/soundscape, also MIT). Read at `main`.
- Licence: MIT, Copyright (c) Microsoft Corporation. Both repositories ship the same licence text.
- What we took: the structure of `apps/ios/GuideDogs/Code/Audio/AudioEngine.swift` and
  `PreparableAudioLayer.swift`: a mono `AVAudioPlayerNode` per source, rendered with HRTFHQ
  through an `AVAudioEnvironmentNode` into the main mixer, the source placed from a bearing,
  the listener at the origin facing -z. `CameraAccess/MetaSight/SpatialBeeper.swift` is a small
  rewrite of that idea (one source, synthesized beeps, no beacons, no geo). Soundscape's engine
  itself (1,300 lines with its settings, sound catalogue and location model) is not vendored.

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
