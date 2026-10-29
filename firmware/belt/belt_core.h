// Belt firmware core: packet parsing, pulse patterns and the link watchdog. Pure C, no hardware,
// so the same file runs on the ESP32 (belt.ino) and on the host (emulator/emu.c).
//
// Command packet, server to belt, UDP, little endian, 26 bytes:
//   'M' 'S' | version (1) | flags | seq u16 | t_ms u32 | 8 x (mode u8, intensity u8)
//   flags bit 0: drop-off ahead (front motor plays the drop pattern over everything else)
//   flags bit 1: arrived at the item being guided to (a sweep once round the belt)
//   mode: 0 off, 1 slow, 2 fast, 3 solid, 4 unknown (no buzz; the sector has no trusted data),
//         5 guide (soft double tap toward a waypoint or a remembered item, panned over two motors)
//         6 find (short sharp ticks toward the item being looked for; intensity = closeness)
//         7 talk (a burst of three soft taps toward whoever is speaking)
//   flags bits 2-3: the activity mode (0 guide, 1 find, 2 talk, 3 explore). When it changes the
//         belt plays that mode's 1.2 s signature on its own clock (see belt_signature below).
// Telemetry, belt to server, 50 Hz, 22 bytes:
//   'M' 'T' | lost u8 | flags u8 | seq u16 (last applied) | t_ms u32 (echo) | rx_age_ms u32 |
//   8 x duty u8 (what the motors are doing right now)
//
// Sectors clockwise from straight ahead: front, front-right, right, back-right, back,
// back-left, left, front-left.
#pragma once
#include <stdint.h>
#include <string.h>

#define BELT_N 8
#define BELT_CMD_LEN 26
#define BELT_TEL_LEN 22
#define BELT_WATCHDOG_MS 250  // no command for this long means the link is lost

enum { M_OFF = 0, M_SLOW, M_FAST, M_SOLID, M_UNKNOWN, M_GUIDE, M_FIND, M_TALK };
#define BELT_SIG_MS 1200

typedef struct {
  uint8_t mode[BELT_N], inten[BELT_N], flags;
  uint16_t seq;
  uint32_t echo_ms, last_rx_ms, sig_ms;
  uint8_t smode;  // the activity mode last seen in the flags
  int have, sig;
} belt_t;

static inline void belt_init(belt_t *b) { memset(b, 0, sizeof *b); }

static inline uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static inline uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

// Returns 1 if the packet was applied. Old or malformed packets are dropped.
static inline int belt_rx(belt_t *b, const uint8_t *p, int n, uint32_t now) {
  if (n != BELT_CMD_LEN || p[0] != 'M' || p[1] != 'S' || p[2] != 1) return 0;
  uint16_t seq = rd16(p + 4);
  if (b->have && (int16_t)(seq - b->seq) <= 0 && now - b->last_rx_ms < BELT_WATCHDOG_MS) return 0;
  uint8_t am = (p[3] >> 2) & 3;
  if (!b->have) b->smode = am;  // the first packet is not a change
  else if (am != b->smode) {
    b->smode = am;
    b->sig = 1;
    b->sig_ms = now;
  }
  b->flags = p[3];
  b->seq = seq;
  b->echo_ms = rd32(p + 6);
  for (int i = 0; i < BELT_N; i++) {
    b->mode[i] = p[10 + 2 * i] <= M_TALK ? p[10 + 2 * i] : M_OFF;
    b->inten[i] = p[11 + 2 * i];
  }
  b->last_rx_ms = now;
  b->have = 1;
  return 1;
}

static inline int belt_lost(const belt_t *b, uint32_t now) {
  return !b->have || now - b->last_rx_ms > BELT_WATCHDOG_MS;
}

// Mode-change signatures, 1.2 s each, strength 180, one pattern per activity mode. They play over
// everything except a lost link, so a mode change is felt before the next cue.
//   guide   two taps then a firm hold on the front motor: "ahead"
//   find    a sweep left to right across the front and back: "scanning"
//   talk    left, right, left, right short taps: "two voices"
//   explore opposite pairs one after another, front/back, right/left, diagonals: "all around"
static inline void belt_signature(uint8_t mode, uint32_t t, uint8_t duty[BELT_N]) {
  for (int i = 0; i < BELT_N; i++) duty[i] = 0;
  switch (mode & 3) {
    case 0:
      if (t < 80 || (t >= 160 && t < 240) || (t >= 400 && t < 800)) duty[0] = 180;
      break;
    case 1: {
      static const uint8_t order[10] = {6, 7, 0, 1, 2, 2, 1, 0, 7, 6};
      if (t < 1200) duty[order[t / 120]] = 180;
      break;
    }
    case 2:
      if (t < 100 || (t >= 400 && t < 500)) duty[6] = 180;
      if ((t >= 200 && t < 300) || (t >= 600 && t < 700)) duty[2] = 180;
      break;
    default: {
      uint32_t k = t / 300;  // 0..3, each pair for 200 ms of its 300 ms
      if (k < 4 && (t % 300) < 200) {
        duty[k] = 180;
        duty[k + 4] = 180;
      }
    }
  }
}

// Motor duty (0..255) for every sector at time `now`. Pulses run on the belt's own clock, so
// network jitter never changes their rhythm.
static inline void belt_tick(const belt_t *b, uint32_t now, uint8_t duty[BELT_N]) {
  if (belt_lost(b, now)) {
    // Lost link: every motor, two short taps every 1.5 s. Never silence.
    uint32_t t = now % 1500;
    uint8_t on = (t < 80 || (t >= 160 && t < 240)) ? 150 : 0;
    for (int i = 0; i < BELT_N; i++) duty[i] = on;
    return;
  }
  if (b->sig && now - b->sig_ms < BELT_SIG_MS) {
    belt_signature(b->smode, now - b->sig_ms, duty);
    return;
  }
  for (int i = 0; i < BELT_N; i++) {
    uint8_t on = 0;
    switch (b->mode[i]) {
      case M_SLOW: on = (now % 667) < 200; break;   // 1.5 Hz, 200 ms taps
      case M_FAST: on = (now % 250) < 125; break;   // 4 Hz
      case M_SOLID: on = 1; break;
      case M_GUIDE: {  // two soft taps a second: unlike any obstacle pattern
        uint32_t t = now % 1000;
        on = t < 70 || (t >= 150 && t < 220);
        break;
      }
      case M_FIND: on = (now % 400) < 40; break;  // 2.5 Hz sharp 40 ms ticks
      case M_TALK: {  // three soft taps, 1.5 s apart
        uint32_t t = now % 1500;
        on = t < 50 || (t >= 130 && t < 180) || (t >= 260 && t < 310);
        break;
      }
      default: on = 0;
    }
    duty[i] = on ? b->inten[i] : 0;
  }
  if (b->flags & 1) {
    // Drop-off ahead: three sharp taps then a pause, full strength, on the front motor.
    uint32_t t = now % 800;
    duty[0] = (t < 300 && (t % 100) < 50) ? 255 : 0;
  }
  if (b->flags & 2) {
    // Arrived: one motor at a time, all the way round, every 960 ms.
    for (int i = 0; i < BELT_N; i++) duty[i] = (now % 960) / 120 == (uint32_t)i ? 200 : 0;
  }
}

static inline void belt_telemetry(const belt_t *b, uint32_t now, const uint8_t duty[BELT_N],
                                  uint8_t out[BELT_TEL_LEN]) {
  uint32_t age = b->have ? now - b->last_rx_ms : 0xffffffffu;
  out[0] = 'M';
  out[1] = 'T';
  out[2] = (uint8_t)belt_lost(b, now);
  out[3] = b->flags;
  out[4] = b->seq & 0xff;
  out[5] = b->seq >> 8;
  for (int k = 0; k < 4; k++) out[6 + k] = (b->echo_ms >> (8 * k)) & 0xff;
  for (int k = 0; k < 4; k++) out[10 + k] = (age >> (8 * k)) & 0xff;
  memcpy(out + 14, duty, BELT_N);
}
