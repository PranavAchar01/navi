// Finds which motor ports have a module on them, without driving anything. Each pin is read with
// the ESP32's own ~45k pull-up, then its pull-down. Nothing attached: the pin follows the pull
// (up=1, down=0). A module whose input has a resistor to ground reads 0 even against the
// pull-up. ADC pins also report the pulled-up voltage, which shows how strong that resistor is.
// A module with a high-impedance input (a comparator chip) looks the same as an empty port.
static const int PINS[] = {13, 12, 14, 27, 26, 25, 33, 32, 4, 16, 17, 18, 19, 21, 22, 23, 5, 2, 15};
static const bool ADC[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1};

void setup() { Serial.begin(115200); }

void loop() {
  Serial.println("---- pin  pullup  pulldown  mV@pullup  verdict");
  for (int i = 0; i < (int)(sizeof(PINS) / sizeof(PINS[0])); i++) {
    int p = PINS[i];
    pinMode(p, INPUT_PULLUP);
    delay(5);
    int up = digitalRead(p);
    int mv = ADC[i] ? analogReadMilliVolts(p) : -1;
    pinMode(p, INPUT_PULLDOWN);
    delay(5);
    int down = digitalRead(p);
    pinMode(p, INPUT);
    const char *v = (up == 1 && down == 0) ? "empty (or high-impedance input)"
                    : (up == 0)            ? "MODULE: pulled to ground"
                                           : "pulled high by something";
    Serial.printf("GPIO %2d     %d       %d      %5d     %s\n", p, up, down, mv, v);
  }
  delay(3000);
}
