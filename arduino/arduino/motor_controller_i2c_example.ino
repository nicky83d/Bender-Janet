#include <Wire.h>

// Janet Arduino I2C motor controller example.
// Janet sends command byte plus two duration bytes little-endian.
// Address: 0x08
// Commands: 0 stop, 1 forward, 2 backward, 3 left, 4 right

const int I2C_ADDR = 0x08;

void setup() {
  Wire.begin(I2C_ADDR);
  Wire.onReceive(receiveEvent);
  // TODO: set your motor driver pins here
}

void loop() {}

void allStop() {
  // TODO: stop motors
}

void driveForward() { /* TODO */ }
void driveBackward() { /* TODO */ }
void turnLeft() { /* TODO */ }
void turnRight() { /* TODO */ }

void receiveEvent(int bytes) {
  if (bytes < 3) return;
  byte command = Wire.read();
  byte lo = Wire.read();
  byte hi = Wire.read();
  unsigned int durationMs = ((unsigned int)hi << 8) | lo;

  switch(command) {
    case 0x01: driveForward(); break;
    case 0x02: driveBackward(); break;
    case 0x03: turnLeft(); break;
    case 0x04: turnRight(); break;
    default: allStop(); return;
  }
  delay(durationMs);
  allStop();
}
