"""
rpi5_serial_sender.py
Standalone serial output module for sending attendance results
from Raspberry Pi 5 to M5StickC Plus 1.1 via USB Serial.

Usage:
    from rpi5_serial_sender import send_result
    send_result("MATCHED", student_admin_number)
"""

import serial
import time

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200


def send_result(result: str, student_id: str = ""):
    """
    Send attendance result to M5StickC via USB Serial.

    Args:
        result: One of MATCHED, NO_MATCH, ENTRY, EXIT, INCOMPLETE
        student_id: The student admin number (optional for NO_MATCH)

    Message format: "RESULT:STUDENT_ID\n"
    """
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(0.1)
        message = f"{result}:{student_id}\n"
        ser.write(message.encode('utf-8'))
        print(f"[SERIAL] Sent: {message.strip()}")
        ser.close()
    except serial.SerialException as e:
        print(f"[SERIAL ERROR] {e}")


if __name__ == "__main__":
    # Quick test
    print("Testing serial sender...")
    send_result("MATCHED", "2301234A")
    time.sleep(1)
    send_result("NO_MATCH", "")
    time.sleep(1)
    send_result("ENTRY", "2301234A")
    time.sleep(1)
    send_result("EXIT", "2301234A")
    time.sleep(1)
    send_result("INCOMPLETE", "2301234A")
    print("Test complete.")
