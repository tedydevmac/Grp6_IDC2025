#include "CytronMotorDriver.h";
#include "Servo.h";
#include "SoftwareSerial.h";
#include "LCD-I2C.h";
#include "Wire.h";

// servo definitions
Servo claw_servo;
Servo forebar_left_servo;
Servo forebar_right_servo;
Servo camera_servo;

// motor pins
CytronMD motorL(PWM_PWM, 3, 6);
CytronMD motorR(PWM_PWM, 5, 11);

// bluetooth definitions
#define BT_TX 10
#define BT_RX 9

// different part definitions
#define left_ir A0
#define right_ir A1

#define claw_pin 4
#define forebar_left 2
#define forebar_right 7
#define camera_pin 8

#define open 0
#define close 1

#define up 0
#define middle 1
#define down 2

#define bandage 0
#define napkin 1
#define syringe 2

#define BUTTON 2
#define DELAY 100
#define NOTE_G4 392
#define NOTE_C5 523
#define NOTE_G5 784
#define NOTE_C6 1047

#define playBtConnectMelody() playMelody(btConnect, btConnectNoteDurations, 2)

// bluetooh serial setup
SoftwareSerial BTSerial(BT_TX, BT_RX); // Maker UNO RX, TX

// LCD Screen
LCD_I2C lcd(0x27, 16, 2);

// vars
int mainSpeed = 191;
int blackThreshold = 700;

int btConnect[] = {NOTE_G5, NOTE_C6};
int btConnectNoteDurations[] = {12, 8};
int pos = 0;

boolean BTConnect = false;
char inChar;

void turnClaw(int opt) {
  if (opt == open) {
    claw_servo.attach(claw_pin); // Attach before moving
    claw_servo.write(0);
    delay(700); // Give the servo time to reach the position
    claw_servo.detach();
  } else if (opt == close) {
    claw_servo.attach(claw_pin); // Attach before moving
    claw_servo.write(179);
    delay(1200); // Give the servo time to reach the position
  }
}

void liftClaw(int opt) {
  if (opt == up) {
    forebar_left_servo.write(100);
    forebar_right_servo.write(82);
    delay(500); // Give the servo time to reach the position
  } else if (opt == middle) {
    forebar_left_servo.write(80);
    forebar_right_servo.write(102);
    delay(500); // Give the servo time to reach the position
  } else if (opt == down) {
    forebar_left_servo.write(60);
    forebar_right_servo.write(122);
    delay(500); // Give the servo time to reach the position
  }
}

void setup()
{
    pinMode(BUTTON, INPUT_PULLUP);
    Serial.begin(9600);
    BTSerial.begin(9600);
    forebar_left_servo.attach(forebar_left);
    forebar_right_servo.attach(forebar_right);
    camera_servo.attach(camera_pin);
    Wire.begin();
    lcd.begin(&Wire);
    lcd.display();
    lcd.backlight();
    delay(DELAY);
    camera_servo.write(30);
    liftClaw(middle);
    turnClaw(open);
    delay(DELAY);
    Serial.println("medical");
}

bool claw_open = true;
double reset_time = 0;
int current = bandage;
String bandage_detection_1 = "Bandage";
String bandage_detection_2 = "Waiting for data";
String napkin_detection_1 = "Gauze";
String napkin_detection_2 = "Waiting for data";
String syringe_detection_1 = "Syringe";
String syringe_detection_2 = "Waiting for data";

void loop()
{
    inChar = '.';

    if (BTSerial.available())
    {
      inChar = BTSerial.read();
    }

    switch (inChar)
    {
      case 'F':
        motorL.setSpeed(mainSpeed);
        motorR.setSpeed(mainSpeed);
        break;
      case 'B':
        motorL.setSpeed(-mainSpeed);
        motorR.setSpeed(-mainSpeed);
        break;
      case 'R':
        motorL.setSpeed(mainSpeed);
        motorR.setSpeed(-mainSpeed);
        delay(DELAY);
        break;
      case 'L':
        motorL.setSpeed(-mainSpeed);
        motorR.setSpeed(mainSpeed);
        delay(DELAY);
        break;
      case 'T':
        turnClaw(open);
        break;
      case 'X':
        turnClaw(close);
        break;
      case 'O':
        liftClaw(middle);
        break;
      case 'S':
        liftClaw(up);
        break;
      case 'A':
        Serial.println("reset");
        break;
      case '0':
        motorL.setSpeed(0);
        motorR.setSpeed(0);
        break;
      default:
        break;
    }

    if (Serial.available() > 0) {
      String data = Serial.readStringUntil('\n');
      Serial.println(data);
      if (data.startsWith("medical_")) {
        if (data.substring(8).startsWith("bandage")) {
          bandage_detection_1 = data.substring(8);
          if (data.substring(16).startsWith("3")) {
            bandage_detection_2 = "Amt: Normal";
          } else {
            bandage_detection_2 = "Amt: Low";
          }
        } else if (data.substring(8).startsWith("napkin")) {
          napkin_detection_1 = data.substring(8);
          if (data.substring(15).startsWith("3")) {
            napkin_detection_2 = "Gauze amt: Normal";
          } else {
            napkin_detection_2 = "Gauze amt: Low";
          }
        } else if (data.substring(8).startsWith("syringe")) {
          syringe_detection_1 = data.substring(8);
          if (data.substring(16).startsWith("3")) {
            syringe_detection_2 = "Amt: Normal";
          } else {
            syringe_detection_2 = "Amt: Low";
          }
        }
      }
    }

    // Cycling mode: always cycle every 3 seconds
    if (millis() > reset_time) {
      lcd.clear();
      if (current == bandage) {
        lcd.setCursor(0, 0);
        lcd.print(bandage_detection_1);
        lcd.setCursor(0, 1);
        lcd.print(bandage_detection_2);
        current = napkin;
      } else if (current == napkin) {
        lcd.setCursor(0, 0);
        lcd.print(napkin_detection_1);
        lcd.setCursor(0, 1);
        lcd.print(napkin_detection_2);
        current = syringe;
      } else if (current == syringe) {
        lcd.setCursor(0, 0);
        lcd.print(syringe_detection_1);
        lcd.setCursor(0, 1);
        lcd.print(syringe_detection_2);
        current = bandage;
      }
      reset_time = millis() + 3000;
    }
}