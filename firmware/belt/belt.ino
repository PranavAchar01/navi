// meta-sight haptic belt, ESP32 38-pin DevKitC on the OSOYOO shield (docs/BUY-plug-and-play.md).
// Joins Wi-Fi, listens for command packets on UDP 4210, drives 8 vibration modules with LEDC PWM.
// All behaviour lives in belt_core.h; the host emulator runs the same file.
#include <WiFi.h>
#include <WiFiUdp.h>
#include "belt_core.h"
#include "secrets.h"  // WIFI_SSID, WIFI_PASS; not checked in

static const int PINS[BELT_N] = {13, 14, 27, 26, 25, 33, 32, 23};  // front, then clockwise
static belt_t belt;
static WiFiUDP udp;
static IPAddress peer;
static uint16_t peerPort = 0;
static uint32_t lastTel = 0;

void setup() {
  for (int i = 0; i < BELT_N; i++) ledcAttach(PINS[i], 2000, 8);
  belt_init(&belt);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  udp.begin(4210);
}

void loop() {
  uint32_t now = millis();
  int n;
  while ((n = udp.parsePacket()) > 0) {
    uint8_t buf[64];
    int len = udp.read(buf, sizeof buf);
    if (belt_rx(&belt, buf, len, now)) {
      peer = udp.remoteIP();
      peerPort = udp.remotePort();
    }
  }
  uint8_t duty[BELT_N];
  belt_tick(&belt, now, duty);  // lost link (Wi-Fi down included) falls through to the tap pattern
  for (int i = 0; i < BELT_N; i++) ledcWrite(PINS[i], duty[i]);
  if (peerPort && now - lastTel >= 20) {
    uint8_t tel[BELT_TEL_LEN];
    belt_telemetry(&belt, now, duty, tel);
    udp.beginPacket(peer, peerPort);
    udp.write(tel, sizeof tel);
    udp.endPacket();
    lastTel = now;
  }
  delay(2);
}
