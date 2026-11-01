// The package ships no types; only the parts splat.ts uses.
declare module "@mkkellogg/gaussian-splats-3d" {
  type Vec3 = [number, number, number];
  export const SceneFormat: { Ply: number };
  export class Viewer {
    constructor(opts: {
      cameraUp?: Vec3;
      initialCameraPosition?: Vec3;
      initialCameraLookAt?: Vec3;
      sharedMemoryForWorkers?: boolean;
    });
    addSplatScene(
      path: string,
      opts: {
        format?: number;
        showLoadingUI?: boolean;
        progressiveLoad?: boolean;
      },
    ): Promise<void>;
    start(): void;
  }
}
