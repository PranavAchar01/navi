// Desk check of the belt's four-motor blend without Bluetooth: walks the eight directions, front
// then clockwise, 1.5 s each with 1 s gaps, using the same diagonal mapping as firmware/belt.
static const int ALL[] = {13, 14, 27, 26, 25, 33, 32, 4, 16, 17, 18, 19, 21, 22, 23};
static const int PINS[4] = {26, 14, 13, 4};  // front-right, back-right, back-left, front-left
static const char *DIRS[8] = {"front", "front-right", "right", "back-right",
                              "back", "back-left", "left", "front-left"};

void setup() {
  Serial.begin(115200);
  for (int p : ALL) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
}

void loop() {
  for (int dir = 0; dir < 8; dir++) {
    Serial.printf("%s:", DIRS[dir]);
    for (int m = 0; m < 4; m++) {
      int c = 2 * m + 1;
      bool on = dir == c || dir == c - 1 || dir == (c + 1) % 8;
      digitalWrite(PINS[m], on ? HIGH : LOW);
      if (on) Serial.printf(" D%d", PINS[m]);
    }
    Serial.println();
    delay(1500);
    for (int p : PINS) digitalWrite(p, LOW);
    delay(1000);
  }
  delay(2000);
}
