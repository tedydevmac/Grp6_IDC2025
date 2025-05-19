#include "CytronMotorDriver.h"
#include <Servo.h>
#include "SoftwareSerial.h"

// servo definitions
Servo claw_servo;
Servo forebar_left_servo;
Servo forebar_right_servo;
Servo camera_servo;

// motor pins
CytronMD motorL(PWM_PWM, 3, 9);  // left
CytronMD motorR(PWM_PWM, 5, 11); // right

// bluetooth definitions
#define BT_TX 6
#define BT_RX 7

// different part definitions
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

#define BUTTON 2
#define DELAY 400
#define NOTE_G4 392
#define NOTE_C5 523
#define NOTE_G5 784
#define NOTE_C6 1047

#define playBtConnectMelody() playMelody(btConnect, btConnectNoteDurations, 2)

// bluetooh serial setup
SoftwareSerial BTSerial(BT_TX, BT_RX); // Maker UNO RX, TX

// vars
int mainSpeed = 191;
int blackThreshold = 700;

int btConnect[] = {NOTE_G5, NOTE_C6};
int btConnectNoteDurations[] = {12, 8};
int pos = 0;

boolean BTConnect = false;
char inChar;

void turnClaw(int opt)
{
    if (opt == open)
    {
        claw_servo.attach(claw_pin); // Attach before moving
        claw_servo.write(0);
        delay(1000); // Give the servo time to reach the position
        claw_servo.detach();
        Serial.println("Opened");
    }
    else if (opt == close)
    {
        claw_servo.attach(claw_pin); // Attach before moving
        claw_servo.write(180);
        delay(1000); // Give the servo time to reach the position
        claw_servo.detach();
        Serial.println("Closed");
    }
}

void lift(int opt)
{
    if (opt == up)
    {
        forebar_left_servo.write(100);
        forebar_right_servo.write(82);
        delay(500); // Give the servo time to reach the position
        Serial.println("Lifted");
    }
    else if (opt == down)
    {
        forebar_left_servo.write(60);
        forebar_right_servo.write(122);
        delay(500); // Give the servo time to reach the position
        Serial.println("Lowered");
    }
}

void setup()
{
    pinMode(BUTTON, INPUT_PULLUP);
    Serial.begin(9600);
    BTSerial.begin(9600);
    claw_servo.attach(claw_pin);
    forebar_left_servo.attach(forebar_left);
    forebar_right_servo.attach(forebar_right);
    camera_servo.attach(camera_pin);
    myservo.attach(9);
    myservo.write(2);
    delay(DELAY);
}
void loop()
{
    inChar = 'S';

    if (BTSerial.available())
    {
        inChar = BTSerial.read();
        Serial.println(inChar);
    }

    if (inChar == 'F')
    {
        motorL.setSpeed(mainSpeed);
        motorR.setSpeed(mainSpeed);
    }
    else if (inChar == 'B')
    {
        motorL.setSpeed(-mainSpeed);
        motorR.setSpeed(-mainSpeed);
    }
    else if (inChar == 'R')
    {
        motorL.setSpeed(mainSpeed);
        motorR.setSpeed(-mainSpeed);
        delay(DELAY);
    }
    else if (inChar == 'L')
    {
        motorL.setSpeed(-mainSpeed);
        motorR.setSpeed(mainSpeed);
        delay(DELAY);
    }
    else if (inChar == 'T')
    {
        turnClaw(open);
    }
    else if (inChar == 'X')
    {
        turnClaw(close);
    }
    else if (inChar == 'C')
    {
        lift(up);
    }
    else if (inChar == 'S')
    {
        lift(down);
    }
}