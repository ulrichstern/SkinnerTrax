/*
  Control LEDs based on commands received from real-time tracker

  12 Aug 2015 by Ulrich Stern

  note:
   send, e.g., "3:128\n" via serial to analogWrite() 128 to pin 3
*/

#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

const long RATE = 115200;   // 115200 or 57600

// do not change
int CNTRLLR_PINS[] = {3, 9, 10, 11};   // TL, TR, BL, BR
const int LED_PIN = 13;
const long SEC = 1000;
const char ITEM_SEP = ':', EOM = '\n';
const boolean BLINK_DURING_SETUP = false;

void setup() {
  Serial.begin(RATE);
  while (!Serial) { }

  for (int i=0; i<NELEMS(CNTRLLR_PINS); i++) {
    int p = CNTRLLR_PINS[i];
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }

  pinMode(LED_PIN, OUTPUT);
  if (BLINK_DURING_SETUP) {
    for (int i=1; i<7; i++) {
      if (i > 1)
        delay(.4*SEC);
      digitalWrite(LED_PIN, i%2);
    }
  }
}

void loop() {
  if (Serial.available() > 0) {
    int pin = Serial.readStringUntil(ITEM_SEP).toInt();
    String valS = Serial.readStringUntil(EOM);
    Serial.print(valS + EOM);
    int val = valS.toInt();
    if (pin == LED_PIN)
      digitalWrite(pin, val);
    else
      analogWrite(pin, val);
  }
}

