/*
  Control Disco board based on commands received from real-time tracker

  19 Jan 2017 by Ulrich Stern

  notes:
  * Disco board has 80 LEDs with 80-channel Teensy- and TLC59711-based LED
   control
  * send, e.g., "1:32768\n" via serial to set the LEDs for chamber 1 to 50% 
   note: TLC59711 is 16 bit
   chambers: 0-19: left side, 20-39: right side; order: like words on page
   to set only top or bottom LED: postfix chamber with 't' or 'b'
*/

#include <Tlc59711.h>

const long RATE = 115200;   // Teensy's USB virtual serial is always 12 Mbit/s
const int NUM_TLC = 8;

// do not change
const char ITEM_SEP = ':', EOM = '\n';

const int EN_PIN = 0, PG_PIN = 1;

Tlc59711 tlc(NUM_TLC);

void setup() {
  setupLxdc();
  tlc.beginFast(false);   // unbuffered faster on Teensy
  tlc.setBrightness(64, 64, 64);
    // max. brightness (127) for Disco board corresponds to 40 mA
  tlc.write();   // for Teensy restart without Disco power-on
  Serial.begin(RATE);
  while (!Serial) { }
}

void loop() {
  if (readCommands())
    tlc.write();
}

// - - - LXDC

void setupLxdc() {
  pinMode(EN_PIN, OUTPUT);
  pinMode(PG_PIN, INPUT_PULLUP);
  powerOnLxdc();
}

void powerOnLxdc() {
  digitalWrite(EN_PIN, HIGH);
  while (!digitalRead(PG_PIN)) { }
}

// - - -

boolean readCommands() {
  boolean didRead = false;
  while (Serial.available() > 0) {
    String chmS = Serial.readStringUntil(ITEM_SEP);
    String valS = Serial.readStringUntil(EOM);
    Serial.print(valS + EOM);
    Serial.send_now();   // transmit partially filled buffer immediately
    uint16_t chm = chmS.toInt();
    char tb = chmS.charAt(chmS.length()-1);
    uint16_t rw = chm / 5, cl = chm % 5, val = valS.toInt();
    uint16_t ch = 12*(rw<2 ? rw : (rw<4 ? rw+4 : rw-2)) + (cl<<1);
    for (int i=0; i<2; i++) {
      if (i == 0 && tb == 'b' || i == 1 && tb == 't')
        continue;
      tlc.setChannel(ch+i, val);
    }
    didRead = true;
  }
  return didRead;
}

