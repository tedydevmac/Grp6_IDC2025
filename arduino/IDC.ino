// Run in Arduino IDE

#include "CytronMotorDriver.h";

#define left_ir A0
#define right_ir A1

#define right 0
#define left 1

#define timing 0
#define junction 1
  
CytronMD motorL(PWM_PWM, 3, 9);
CytronMD motorR(PWM_PWM, 10, 11);

int mainSpeed = 127;
  
void setup() {
  pinMode(left_ir, INPUT);
  pinMode(right_ir, INPUT);
  Serial.begin(9600);
}

void move(int ms, int speed = mainSpeed) {
  motorL.setSpeed(speed);
  motorR.setSpeed(speed);
}

void spotTurn(int ms, int direction, int speed = mainSpeed) {
  if (direction == right) {
    motorL.setSpeed(speed);
    motorR.setSpeed(-speed);
  } else {
    motorL.setSpeed(-speed);
    motorR.setSpeed(speed);
  }
}

void curveTurn(int curve, int speed = mainSpeed) {
  if (curve > 0) {
    motorL.setSpeed(speed);
    motorR.setSpeed(max(speed - curve, 0));
  } else {
    motorL.setSpeed(max(speed + curve, 0));
    motorR.setSpeed(speed);
  }
}

void noobLineTrace() {
  int left_read = analogRead(left_ir);
  int right_read = analogRead(right_ir);

  bool left_white = left_read < 500;
  bool right_white = right_read < 500;

  if (left_white && right_white) {
    move(0);
  } else if (!left_white && right_white) {
    spotTurn(0, left);
  } else if (left_white && !right_white) {
    spotTurn(0, right);
  }
}

void pidLineTrace(int sec, int speed = mainSpeed) {
  float kp = 0.10; // fixed
  float kd = 0.005; // adjust as needed by ±0.005 increments
  float prev_error = 0;
  double stop_time = millis() + sec * 1000;

  int left_read, right_read, error;
  float pid_error;

  while (millis() < stop_time) {
    left_read = analogRead(left_ir);
    right_read = analogRead(right_ir);
    // Serial.print(left_read);
    // Serial.print(" ");
    // Serial.println(right_read);

    error = right_read - left_read;
    pid_error = error * kp + (error - prev_error) * kd;
    // Serial.print(error);
    // Serial.print(" ");
    // Serial.println(pid_error);

    curveTurn(pid_error);

    prev_error = error;
  }
}
  
void loop() {
  pidLineTrace(100);
}