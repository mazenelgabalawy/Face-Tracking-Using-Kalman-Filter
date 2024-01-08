#include <Servo.h>

Servo myservo;

int angle = 90;
int width;
int Desired_center;
int Actual_center = 250;

void setup() {
  Serial.begin(115200);

  myservo.attach(9);
  myservo.write(90);

}

void loop() {
  // put your main code here, to run repeatedly:
  while(Serial.available()==0){

  }

  width = Serial.readStringUntil(',').toInt();
  Desired_center = width/2;
  Actual_center = Serial.readStringUntil('\r').toInt();


  if (Actual_center > Desired_center+15){
    angle = angle - 1;
  }
  else if(Actual_center < Desired_center-15){
    angle = angle + 1;
  }
  else{
    angle = angle;
  }

  if (angle > 180){
    angle = 180;
  }

  if (angle < 0){
    angle = 0;
  }

  // Serial.println(angle);
  myservo.write(angle);


}
