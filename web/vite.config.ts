import { defineConfig } from "vite";

const target = process.env.METASIGHT_BACKEND ?? "http://127.0.0.1:8444";

export default defineConfig({
  server: {
    host: true,
    proxy: {
      "/api": target,
      "/ws": { target, ws: true },
    },
  },
  // Inline PostCSS config so a stray postcss.config in a parent folder is never picked up.
  css: { postcss: { plugins: [] } },
  build: { outDir: "dist", sourcemap: false },
});
