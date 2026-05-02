// CubeMars AK45-10 — enter MIT mode only
// Hardware: Arduino UNO R4 WiFi
// CAN wiring: CANTX → D4, CANRX → D5, common GND with motor driver

#include <Arduino.h>
#include <Arduino_CAN.h>

void setup() {
    Serial.begin(115200);
    delay(100);
    CAN.begin(CanBitRate::BR_1000k);
    delay(200);
}

void loop() {

    uint8_t enter_cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};
    CanMsg msg(CanStandardId(0x01), 8, enter_cmd);
    CAN.write(msg);
    
    Serial.println("Sent MIT enter mode");
    delay(200);

}
