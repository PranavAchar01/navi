// The room as Gaussian splats, trained from the dense run's frames (scripts/room_splat.py).
// The splat is in the room frame: y up, floor at 0, same as room.ply and the saved objects.
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

const status = document.getElementById("status") as HTMLDivElement;

const viewer = new GaussianSplats3D.Viewer({
  cameraUp: [0, 1, 0],
  // Where the first training frame stood, by the bed, facing the whiteboard and shelves.
  initialCameraPosition: [1.23, 1.02, 0.14],
  initialCameraLookAt: [1.21, 0.85, -0.81],
  sharedMemoryForWorkers: false, // the server sends no cross-origin isolation headers
});

viewer
  .addSplatScene("/api/splat.ply", {
    format: GaussianSplats3D.SceneFormat.Ply,
    showLoadingUI: false,
    progressiveLoad: true,
  })
  .then(() => {
    status.textContent = "Drag to look around, scroll to move";
    viewer.start();
  })
  .catch(() => {
    status.textContent =
      "No room splat yet (run scripts/room_splat.py, then Brush)";
  });
