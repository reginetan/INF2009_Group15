/*
 * m5stickc_attendance.ino
 * M5StickC Plus 1.1 - Attendance Verification Display
 * 
 * Receives serial messages from RPi5 via USB-C.
 * Format: "RESULT:STUDENT_ID\n"
 * 
 * Results:
 *   MATCHED    → GREEN screen,  2 ascending beeps
 *   NO_MATCH   → RED screen,    1 long low beep
 *   ENTRY      → BLUE screen,   1 short beep
 *   EXIT       → ORANGE screen, 2 descending beeps
 *   INCOMPLETE → RED screen,    3 warning beeps
 * 
 * Buzzer: GPIO 2
 * Serial: 115200 baud
 */

#include <M5StickCPlus.h>

#define BUZZER_PIN 2

// Colours
#define CLR_GREEN   0x07E0
#define CLR_RED     0xF800
#define CLR_BLUE    0x001F
#define CLR_ORANGE  0xFD20
#define CLR_BLACK   0x0000
#define CLR_WHITE   0xFFFF

String inputBuffer = "";

// ─── Forward declarations ───────────────────────────────────────
void handleMessage(String result, String studentId);
void showScreen(uint16_t bg, const char* line1, const char* line2, const char* line3);
void beepAscending();
void beepDescending();
void beepLongLow();
void beepShort();
void beepWarning();
void toneCustom(int freq, int durationMs);

// ─── Setup ──────────────────────────────────────────────────────
void setup() {
    M5.begin();
    Serial.begin(115200);

    // Buzzer pin
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    // LCD setup
    M5.Lcd.setRotation(1);
    M5.Lcd.fillScreen(CLR_BLACK);
    M5.Lcd.setTextColor(CLR_WHITE);
    M5.Lcd.setTextSize(2);
    M5.Lcd.setCursor(10, 30);
    M5.Lcd.println("ATTENDANCE");
    M5.Lcd.setCursor(10, 60);
    M5.Lcd.println("SYSTEM READY");
    M5.Lcd.setTextSize(1);
    M5.Lcd.setCursor(10, 100);
    M5.Lcd.println("Waiting for RPi5...");
}

// ─── Loop ───────────────────────────────────────────────────────
void loop() {
    M5.update();

    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n') {
            inputBuffer.trim();
            if (inputBuffer.length() > 0) {
                // Parse "RESULT:STUDENT_ID"
                int colonIdx = inputBuffer.indexOf(':');
                String result = "";
                String studentId = "";

                if (colonIdx >= 0) {
                    result = inputBuffer.substring(0, colonIdx);
                    studentId = inputBuffer.substring(colonIdx + 1);
                } else {
                    result = inputBuffer;
                }

                handleMessage(result, studentId);
            }
            inputBuffer = "";
        } else {
            inputBuffer += c;
        }
    }
}

// ─── Handle parsed message ──────────────────────────────────────
void handleMessage(String result, String studentId) {
    result.trim();
    studentId.trim();

    if (result == "MATCHED") {
        showScreen(CLR_GREEN, "MATCHED", studentId.c_str(), "");
        beepAscending();
    }
    else if (result == "NO_MATCH") {
        showScreen(CLR_RED, "NO MATCH", "", "");
        beepLongLow();
    }
    else if (result == "ENTRY") {
        showScreen(CLR_BLUE, "ENTRY", "Logged IN", studentId.c_str());
        beepShort();
    }
    else if (result == "EXIT") {
        showScreen(CLR_ORANGE, "EXIT", "Logged OUT", studentId.c_str());
        beepDescending();
    }
    else if (result == "INCOMPLETE") {
        showScreen(CLR_RED, "INCOMPLETE", "Flag for review", studentId.c_str());
        beepWarning();
    }
    else {
        showScreen(CLR_BLACK, "UNKNOWN CMD", result.c_str(), "");
    }
}

// ─── Display helper ─────────────────────────────────────────────
void showScreen(uint16_t bg, const char* line1, const char* line2, const char* line3) {
    M5.Lcd.fillScreen(bg);

    // Choose readable text colour: white for dark backgrounds, black for bright
    uint16_t textClr = CLR_WHITE;
    if (bg == CLR_GREEN || bg == CLR_ORANGE) {
        textClr = CLR_BLACK;
    }
    M5.Lcd.setTextColor(textClr);

    // Line 1 – main status (large)
    M5.Lcd.setTextSize(3);
    M5.Lcd.setCursor(10, 20);
    M5.Lcd.println(line1);

    // Line 2 – sub-status (medium)
    M5.Lcd.setTextSize(2);
    M5.Lcd.setCursor(10, 65);
    M5.Lcd.println(line2);

    // Line 3 – student ID (small)
    M5.Lcd.setTextSize(2);
    M5.Lcd.setCursor(10, 100);
    M5.Lcd.println(line3);
}

// ─── Buzzer patterns ────────────────────────────────────────────

void toneCustom(int freq, int durationMs) {
    long period = 1000000L / freq;
    long halfPeriod = period / 2;
    long cycles = (long)freq * durationMs / 1000;

    for (long i = 0; i < cycles; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        delayMicroseconds(halfPeriod);
        digitalWrite(BUZZER_PIN, LOW);
        delayMicroseconds(halfPeriod);
    }
}

// MATCHED: 2 ascending beeps
void beepAscending() {
    toneCustom(1000, 120);
    delay(80);
    toneCustom(1500, 120);
}

// NO_MATCH: 1 long low beep
void beepLongLow() {
    toneCustom(400, 500);
}

// ENTRY: 1 short beep
void beepShort() {
    toneCustom(1200, 100);
}

// EXIT: 2 descending beeps
void beepDescending() {
    toneCustom(1500, 120);
    delay(80);
    toneCustom(1000, 120);
}

// INCOMPLETE: 3 warning beeps
void beepWarning() {
    for (int i = 0; i < 3; i++) {
        toneCustom(800, 150);
        delay(100);
    }
}
