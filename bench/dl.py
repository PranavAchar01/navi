from huggingface_hub import snapshot_download, hf_hub_download
import sys
for repo in ["facebook/map-anything-apache","facebook/VGGT-1B","depth-anything/Depth-Anything-V2-Small-hf","depth-anything/Depth-Anything-V2-Base-hf","apple/DepthPro-hf"]:
    try:
        p=snapshot_download(repo); print("OK",repo,p,flush=True)
    except Exception as e: print("FAIL",repo,repr(e)[:200],flush=True)
try:
    p=hf_hub_download("lch01/StreamVGGT","checkpoints.pth"); print("OK streamvggt",p,flush=True)
except Exception as e: print("FAIL streamvggt",repr(e)[:300],flush=True)
