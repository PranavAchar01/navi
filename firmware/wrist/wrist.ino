// meta-sight wrist Braille band, classic ESP32 38-pin DevKitC. Six DiGiYes/DAOKAI vibration
// modules (IN/VCC/GND, high = buzz), one per Braille dot. Prototype: not yet run on hardware.
// Takes one BLE characteristic write per cell (see braille_core.h). The belt protocol is untouched.
//
//   dot 1 -> GPIO 4    dot 4 -> GPIO 18      cell:   1 o o 4     left column dots 1-3,
//   dot 2 -> GPIO 16   dot 5 -> GPIO 19              2 o o 5     right column dots 4-6,
//   dot 3 -> GPIO 17   dot 6 -> GPIO 21              3 o o 6     worn with dot 1 toward the hand
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include "braille_core.h"

// Must match WRIST_SERVICE_UUID / WRIST_CHAR_UUID in metasight/haptics.py.
#define SERVICE_UUID "6d657461-7369-6768-742d-627261696c6c"
#define CHAR_UUID "6d657461-7369-6768-742d-627261696c01"

static const int PINS[WRIST_N] = {4, 16, 17, 18, 19, 21};
static wrist_t wrist;

class CellCallback : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    auto v = c->getValue();  // String on core 3.x, std::string on 2.x
    wrist_rx(&wrist, (const uint8_t *)v.c_str(), (int)v.length(), millis());
  }
};

class Reconnect : public BLEServerCallbacks {
  void onDisconnect(BLEServer *s) override {
    wrist.mask = 0;  // a dropped link never leaves a motor running
    s->startAdvertising();
  }
};

void setup() {
  for (int i = 0; i < WRIST_N; i++) {
    pinMode(PINS[i], OUTPUT);
    digitalWrite(PINS[i], LOW);
  }
  wrist_init(&wrist);
  BLEDevice::init("metasight-wrist");
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new Reconnect());
  BLEService *svc = server->createService(SERVICE_UUID);
  BLECharacteristic *ch = svc->createCharacteristic(
      CHAR_UUID, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  ch->setCallbacks(new CellCallback());
  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->start();
}

void loop() {
  uint8_t m = wrist_tick(&wrist, millis());
  for (int i = 0; i < WRIST_N; i++) digitalWrite(PINS[i], (m >> i) & 1);
  delay(2);
}
