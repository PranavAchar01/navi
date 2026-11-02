// meta-sight haptic belt, ESP32 38-pin DevKitC on the OSOYOO shield (docs/BUY-plug-and-play.md).
// Takes the 26-byte command packet from belt_core.h over Bluetooth LE (the phone app relays it)
// and, when a secrets.h exists next to this file, also over Wi-Fi UDP 4210 (the server direct).
// The same packet also works over the USB cable (115200 baud), with telemetry back on it while
// the cable is talking (desk tests and the sim driving the real motors).
// All feed the same belt_core: 8 vibration modules on LEDC PWM, and the 250 ms lost-link taps.
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include "belt_core.h"

#if __has_include("secrets.h")
#include <WiFi.h>
#include <WiFiUdp.h>
#include "secrets.h"  // WIFI_SSID, WIFI_PASS; not checked in
#define BELT_WIFI 1
#endif

// Must match BELT_SERVICE_UUID / BELT_CHAR_UUID in metasight/belt.py and ios .../BeltBLE.swift.
#define BELT_SERVICE_UUID "6d657461-7369-6768-742d-62656c740000"
#define BELT_CHAR_UUID "6d657461-7369-6768-742d-62656c740001"
#define BELT_BLE_NAME "metasight-belt"
#define BELT_BLE_MTU 185  // the 26-byte packet needs more than the default 23-byte MTU

// Four modules stand in for the eight directions: each sits on a diagonal (front-right, back-right,
// back-left, front-left). Its own direction plays at full strength and each direction beside it
// at 1/sqrt(2) (constant power), so "front" buzzes both front motors at 71% and "front-right"
// only the one; the pair feels as strong as a single motor. Order: cell 1, 3, 5, 7.
#define MOTORS 4
static const int PINS[MOTORS] = {26, 14, 13, 4};  // front-right, back-right, back-left, front-left
static belt_t belt;

// BLE writes arrive on the Bluetooth task; the latest one waits here for loop().
static portMUX_TYPE rxMux = portMUX_INITIALIZER_UNLOCKED;
static uint8_t pending[BELT_CMD_LEN];
static volatile bool havePending = false;

class CommandCallback : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    auto v = c->getValue();  // String on core 3.x, std::string on 2.x
    if (v.length() != BELT_CMD_LEN) return;
    portENTER_CRITICAL(&rxMux);
    memcpy(pending, v.c_str(), BELT_CMD_LEN);
    havePending = true;
    portEXIT_CRITICAL(&rxMux);
  }
};

class Reconnect : public BLEServerCallbacks {
  void onDisconnect(BLEServer *s) override {
    s->startAdvertising();  // the watchdog plays the lost taps meanwhile
  }
};

static void startBle() {
  BLEDevice::init(BELT_BLE_NAME);
  BLEDevice::setMTU(BELT_BLE_MTU);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new Reconnect());
  BLEService *svc = server->createService(BELT_SERVICE_UUID);
  BLECharacteristic *ch = svc->createCharacteristic(
      BELT_CHAR_UUID, BLECharacteristic::PROPERTY_WRITE_NR | BLECharacteristic::PROPERTY_WRITE);
  ch->setCallbacks(new CommandCallback());
  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(BELT_SERVICE_UUID);
  adv->setScanResponse(true);
  adv->start();
}

// USB serial: packets are found by their 'M' 'S' header; telemetry goes back while one arrived
// in the last second.
static uint8_t serBuf[BELT_CMD_LEN];
static int serLen = 0;
static uint32_t serLast = 0, serTel = 0;

static void pollSerial(uint32_t now) {
  while (Serial.available()) {
    uint8_t c = Serial.read();
    if ((serLen == 0 && c != 'M') || (serLen == 1 && c != 'S')) {
      serLen = c == 'M';
      if (serLen) serBuf[0] = c;
      continue;
    }
    serBuf[serLen++] = c;
    if (serLen == BELT_CMD_LEN) {
      belt_rx(&belt, serBuf, BELT_CMD_LEN, now);  // drops a misaligned or stale one
      serLast = now;
      serLen = 0;
    }
  }
}

#ifdef BELT_WIFI
static WiFiUDP udp;
static IPAddress peer;
static uint16_t peerPort = 0;
static uint32_t lastTel = 0;
#endif

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < MOTORS; i++) ledcAttach(PINS[i], 2000, 8);
  belt_init(&belt);
  startBle();
#ifdef BELT_WIFI
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  udp.begin(4210);
#endif
}

void loop() {
  uint32_t now = millis();
  if (havePending) {
    uint8_t buf[BELT_CMD_LEN];
    portENTER_CRITICAL(&rxMux);
    memcpy(buf, pending, BELT_CMD_LEN);
    havePending = false;
    portEXIT_CRITICAL(&rxMux);
    belt_rx(&belt, buf, BELT_CMD_LEN, now);
  }
  pollSerial(now);
#ifdef BELT_WIFI
  while (udp.parsePacket() > 0) {
    uint8_t buf[64];
    int len = udp.read(buf, sizeof buf);
    if (belt_rx(&belt, buf, len, now)) {
      peer = udp.remoteIP();
      peerPort = udp.remotePort();
    }
  }
#endif
  uint8_t duty[BELT_N];
  belt_tick(&belt, now, duty);  // lost link falls through to the tap pattern
  for (int m = 0; m < MOTORS; m++) {
    int c = 2 * m + 1;
    float a = duty[c], l = duty[c - 1], r = duty[(c + 1) % BELT_N];
    ledcWrite(PINS[m], (uint8_t)min(255.0f, sqrtf(a * a + 0.5f * (l * l + r * r))));
  }
  if (serLast && now - serLast < 1000 && now - serTel >= 20) {
    uint8_t tel[BELT_TEL_LEN];
    belt_telemetry(&belt, now, duty, tel);
    Serial.write(tel, sizeof tel);
    serTel = now;
  }
#ifdef BELT_WIFI
  if (peerPort && now - lastTel >= 20) {
    uint8_t tel[BELT_TEL_LEN];
    belt_telemetry(&belt, now, duty, tel);
    udp.beginPacket(peer, peerPort);
    udp.write(tel, sizeof tel);
    udp.endPacket();
    lastTel = now;
  }
#endif
  delay(2);
}
