/*
  Control TLC59711-based LED controller based on commands received from
  real-time tracker

  21 Jan 2016 by Ulrich Stern

  note:
   send, e.g., "1:32768\n" via serial to set LED 1 to 50% (TLC59711 is 16 bit)
    mapping: 0:R0 (chip 1), 1:G0, 2:B0, 3:R1, ..., 12:R0 (chip 2), ...
*/

#include <Tlc59711.h>

const long RATE = 115200;   // 115200 or 57600
const int NUM_TLC = 2;

// do not change
const char ITEM_SEP = ':', EOM = '\n';

Tlc59711 tlc(NUM_TLC);

void setup() {
  Serial.begin(RATE);
  tlc.beginFast();
  tlc.write();
}

void loop() {
  if (readCommands())
    tlc.write();
}

boolean readCommands() {
  boolean didRead = false;
  while (Serial.available() > 0) {
    int idx = Serial.readStringUntil(ITEM_SEP).toInt();
    String valS = Serial.readStringUntil(EOM);
    Serial.print(valS + EOM);
    tlc.setChannel(idx, valS.toInt());
    didRead = true;
  }
  return didRead;
}

