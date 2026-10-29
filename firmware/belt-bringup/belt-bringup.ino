// meta-sight belt wiring check: each motor buzzes once, in order (front, then clockwise), about
// 8 s a round, for 7 rounds (about a minute). The serial monitor (115200) names each motor.
// Flash with: scripts/flash_belt.sh bringup. Flash the real belt firmware afterwards.
#include <Arduino.h>

static const int PINS[8] = {13, 14, 27, 26, 25, 33, 32, 23};
static const char *NAMES[8] = {"front",     "front-right", "right",      "back-right",
                               "back",      "back-left",   "left",       "front-left"};
static const int ROUNDS = 7;

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 8; i++) {
    pinMode(PINS[i], OUTPUT);
    digitalWrite(PINS[i], LOW);
  }
  delay(1000);
  for (int r = 1; r <= ROUNDS; r++) {
    Serial.printf("round %d of %d\n", r, ROUNDS);
    for (int i = 0; i < 8; i++) {
      Serial.printf("motor %d %s (GPIO %d)\n", i, NAMES[i], PINS[i]);
      digitalWrite(PINS[i], HIGH);
      delay(400);
      digitalWrite(PINS[i], LOW);
      delay(600);
    }
  }
  Serial.println("done: every motor should have buzzed once per round, front then clockwise");
}

void loop() { delay(1000); }
