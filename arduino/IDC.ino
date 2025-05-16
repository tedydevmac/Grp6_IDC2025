#include "CytronMotorDriver.h";
#include "Servo.h";

#define left_ir A0
#define right_ir A1

#define claw_pin 4
#define forebar_left 6
#define forebar_right 7
#define camera_pin 8

#define left 0
#define right 1
#define middle 2

#define timing 0
#define junction 1

#define open 0
#define close 1

#define up 0
#define down 1

CytronMD motorL(PWM_PWM, 3, 9);
CytronMD motorR(PWM_PWM, 5, 11);

Servo claw_servo;
Servo forebar_left_servo;
Servo forebar_right_servo;
Servo camera_servo;

int mainSpeed = 191;
int blackThreshold = 700;
  
void setup() {
  pinMode(left_ir, INPUT);
  pinMode(right_ir, INPUT);
  claw_servo.attach(claw_pin);
  forebar_left_servo.attach(forebar_left);
  forebar_right_servo.attach(forebar_right);
  camera_servo.attach(camera_pin);
  Serial.begin(9600);
  delay(10);
}

void move(float sec, int speed = mainSpeed) {
  double stop_time = millis() + abs(sec) * 1000;
  if (sec > 0) {
    motorL.setSpeed(speed);
    motorR.setSpeed(speed);
  } else {
    motorL.setSpeed(-speed);
    motorR.setSpeed(-speed);
  }
  while (millis() < stop_time) {}
  motorL.setSpeed(0);
  motorR.setSpeed(0);
}

void spotTurn(float sec, int direction, int speed = mainSpeed) {
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
    // Serial.print(speed);
    // Serial.print(" ");
    // Serial.println(max(speed - curve, 0));
  } else {
    motorL.setSpeed(max(speed + curve, 0));
    motorR.setSpeed(speed);
    // Serial.print(max(speed + curve, 0));
    // Serial.print(" ");
    // Serial.println(speed);
  }
}

void oneSensorLineTrace(int type, int side, float sec = 0, int speed = mainSpeed) {
  float kp = 0.6;
  float kd = -0.005;
  float prev_error = 0;

  int left_read = 0, right_read = 0, error;
  float pid_error;
  if (type == timing) {
    double stop_time = millis() + sec * 1000;

    if (side == right) {
      while (millis() < stop_time) {
        left_read = analogRead(left_ir);
        right_read = analogRead(right_ir);
        // Serial.print(left_read);
        // Serial.print(" ");
        // Serial.println(right_read);

        error = right_read - 400;
        pid_error = error * kp + (error - prev_error) * kd;

        curveTurn(pid_error);

        prev_error = error;
      }
    } else if (side == left) {
      while (millis() < stop_time) {
        left_read = analogRead(left_ir);
        right_read = analogRead(right_ir);
        // Serial.print(left_read);
        // Serial.print(" ");
        // Serial.println(right_read);

        error = 400 - left_read;
        pid_error = error * kp + (error - prev_error) * kd;

        curveTurn(pid_error);

        prev_error = error;
      }
    }
  } else if (type == junction) {
    if (side == right) {
      while (left_read < blackThreshold) {
        left_read = analogRead(left_ir);
        right_read = analogRead(right_ir);
        // Serial.print(left_read);
        // Serial.print(" ");
        // Serial.println(right_read);

        error = right_read - 400;
        pid_error = error * kp + (error - prev_error) * kd;

        curveTurn(pid_error);

        prev_error = error;
      }
    } else if (side == left) {
      while (right_read < blackThreshold) {
        left_read = analogRead(left_ir);
        right_read = analogRead(right_ir);
        // Serial.print(left_read);
        // Serial.print(" ");
        // Serial.println(right_read);

        error = 400 - left_read;
        pid_error = error * kp + (error - prev_error) * kd;

        curveTurn(pid_error);

        prev_error = error;
      }
    }
  }
  move(0,0);
}

void twoSensorLineTrace(int opt, float sec = 0, int speed = mainSpeed) {
  float kp = 0.1;
  float kd = -0.015;
  float prev_error = 0;

  int left_read = 0, right_read = 0, error;
  float pid_error;

  if (opt == timing) {
    double stop_time = millis() + sec * 1000;

    while (millis() < stop_time) {
      left_read = analogRead(left_ir);
      right_read = analogRead(right_ir);
      // Serial.print(left_read);
      // Serial.print(" ");
      // Serial.println(right_read);

      error = right_read - left_read + 80;
      pid_error = error * kp + (error - prev_error) * kd;

      curveTurn(pid_error);

      prev_error = error;
    }
  } else if (opt == junction) {
    while (left_read < blackThreshold || right_read < blackThreshold) {
      left_read = analogRead(left_ir);
      right_read = analogRead(right_ir);
      Serial.print(left_read);
      Serial.print(" ");
      Serial.println(right_read);

      error = right_read - left_read + 80;
      pid_error = error * kp + (error - prev_error) * kd;

      curveTurn(pid_error);

      prev_error = error;
    }
  }
  move(0,0);
}

void turnClaw(int opt) {
  if (opt == open) {
    claw_servo.attach(claw_pin); // Attach before moving
    claw_servo.write(0);
    delay(1000); // Give the servo time to reach the position
    claw_servo.detach();
    Serial.println("Opened");
  } else if (opt == close) {
    claw_servo.attach(claw_pin); // Attach before moving
    claw_servo.write(180);
    delay(1000); // Give the servo time to reach the position
    claw_servo.detach();
    Serial.println("Closed");
  }
}

void lift(int opt) {
  if (opt == up) {
    forebar_left_servo.write(100);
    forebar_right_servo.write(82);
    delay(500); // Give the servo time to reach the position
    Serial.println("Lifted");
  } else if (opt == down) {
    forebar_left_servo.write(60);
    forebar_right_servo.write(122);
    delay(500); // Give the servo time to reach the position
    Serial.println("Lowered");
  }
}

void autoTurn() {
  oneSensorLineTrace(timing, left, 0.5);
  oneSensorLineTrace(junction, left);
  oneSensorLineTrace(timing, left, 0.1);
  twoSensorLineTrace(junction, 0, 64);
}

bool run = true;
int choice1 = "sandwich";
int choice2 = "napkin"; 
int location = middle;
  
void loop() {
  if (run) {
    // Init
    lift(down);
    turnClaw(open);
    // camera_servo.write(90);

    // Testing
    // move(0.5);
    // twoSensorLineTrace(timing, 100);
    // move(-0.5);
    turnClaw(close);
    lift(up);
    motorL.setSpeed(-200);
    motorR.setSpeed(-100);
    delay(1000);
    motorL.setSpeed(0);
    motorR.setSpeed(0);

    // // Go to food
    // move(0.5);
    // oneSensorLineTrace(timing, left, 0.5);
    // autoTurn();

    // // Detect food location
    // Serial.println("food");
    // for(int i = 0; i < 3; i++) {
    //   if (Serial.available() > 0) {
    //     String data = Serial.readStringUntil("\n");
    //     if (data.startsWith("food_") && data.substring(5) == choice1) {
    //       break;
    //     } else {
    //       if (location == middle) {
    //         location = left;
    //         camera_servo.write(60);
    //       } else if (location == left) {
    //         location = right;
    //         camera_servo.write(120);
    //       }
    //     }
    //   }
    // }
    
    // // Get food
    // if (location == middle) {
    //   turnClaw(close);
    //   spotTurn(1);
    // } else if (location == left) {
    //   spotTurn(0.25);
    //   turnClaw(close);
    //   spotTurn(-0.75);
    // } else if (location == right) {
    //   spotTurn(-0.25);
    //   turnClaw(close);
    //   spotTurn(0.75);
    // }


    Serial.println("done");
    run = false;
  }
}