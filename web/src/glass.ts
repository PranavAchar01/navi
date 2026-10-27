/** Liquid glass, the cheap parts. The frosted layers and the rim light are plain CSS and work
 *  everywhere (including iOS Safari). Real refraction (an SVG displacement map applied to the
 *  backdrop) only exists in Chromium, so it is switched on there for small controls only; big
 *  panels stay blur-only so a phone never pays for it. The rim light follows the pointer or the
 *  tilt of the device so the glass reads as a surface, not a sticker. */
export function setupGlass(): void {
  const root = document.documentElement;
  const chromium = !!(navigator as Navigator & { userAgentData?: { brands: { brand: string }[] } })
    .userAgentData?.brands.some((b) => b.brand === "Chromium");
  if (chromium && matchMedia("(hover: hover)").matches) root.classList.add("refract");
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  let ang = 135;
  let pending = false;
  const apply = () => {
    pending = false;
    root.style.setProperty("--ang", `${ang.toFixed(0)}deg`);
  };
  const set = (deg: number) => {
    if (Math.abs(deg - ang) < 2) return;
    ang = deg;
    if (!pending) {
      pending = true;
      requestAnimationFrame(apply);
    }
  };
  addEventListener("pointermove", (e) => {
    const dx = e.clientX / innerWidth - 0.5;
    const dy = e.clientY / innerHeight - 0.5;
    set((Math.atan2(dy, dx) * 180) / Math.PI + 90);
  });
  addEventListener("deviceorientation", (e) => {
    if (e.gamma === null || e.beta === null) return;
    set(135 + e.gamma * 1.4 - (e.beta - 60) * 0.4);
  });
}
