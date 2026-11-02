// Every motor port held low, nothing else: flash this to park the board silently on the desk.
static const int PINS[] = {13, 14, 27, 26, 25, 33, 32, 23, 4, 16, 17, 18, 19, 21};

void setup() {
  for (int p : PINS) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
}

void loop() { delay(1000); }
