"""The safety loop's obstacle detector as a Core ML model for the server.

    .venv/bin/python scripts/coreml_detect.py

yolo26n at the server's 384 input, float16, exported with ultralytics next to the .pt in the
weights folder (yolo26n.mlpackage). On the Mac it leaves the GPU to the map: the PyTorch
detector takes about 7 ms alone and 25 ms while MapAnything runs. coremltools' integer cast fails
on numpy 2 (`int()` of a one-element array), so it is patched for the export only.
"""

import numpy as np
from coremltools.converters.mil.frontend.torch import ops
from ultralytics import YOLO

from metasight.config import Settings

_cast = ops._cast


def _cast_np2(context, node, dtype, dtype_name):
    x = context[node.inputs[0]]
    if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray) and x.val.size == 1:
        context.add(ops.mb.const(val=dtype(x.val.item()), name=node.name))
        return
    _cast(context, node, dtype, dtype_name)


ops._cast = _cast_np2
s = Settings.load()
print(
    YOLO(str(s.weights_dir / "yolo26n.pt")).export(
        format="coreml", imgsz=s.detector_imgsz, half=True
    )
)
