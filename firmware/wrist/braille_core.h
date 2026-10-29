// Wrist Braille band: the logic, free of Arduino so the host test (tests/test_braille.py) builds
// it. A BLE write is 3 bytes: 6-bit dot mask (bit 0 = dot 1 ... bit 5 = dot 6), then the on-time
// in ms as u16 little-endian. The band times the buzz itself, so BLE jitter never stretches a cell.
// Mask 0 or duration 0 switches every motor off. A cell never runs longer than WRIST_MAX_MS.
#pragma once
#include <stdint.h>

#define WRIST_N 6
#define WRIST_MAX_MS 2000
#define WRIST_PKT_LEN 3

typedef struct {
  uint8_t mask;
  uint32_t off_at;
} wrist_t;

static inline void wrist_init(wrist_t *w) {
  w->mask = 0;
  w->off_at = 0;
}

// Returns 1 if the packet was applied.
static inline int wrist_rx(wrist_t *w, const uint8_t *p, int n, uint32_t now) {
  if (n != WRIST_PKT_LEN) return 0;
  uint16_t dur = (uint16_t)(p[1] | p[2] << 8);
  if (dur > WRIST_MAX_MS) dur = WRIST_MAX_MS;
  w->mask = (p[0] & 0x3F);
  if (dur == 0) w->mask = 0;
  w->off_at = now + dur;
  return 1;
}

// Dot mask that should be buzzing at `now`.
static inline uint8_t wrist_tick(wrist_t *w, uint32_t now) {
  if (w->mask && (int32_t)(now - w->off_at) >= 0) w->mask = 0;
  return w->mask;
}
