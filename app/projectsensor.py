from rd03d import RD03D
import time
import paho.mqtt.client as mqtt

# --- MQTT Setup ---
BROKER_IP = "172.20.10.13"  # Replace with your Inference Pi's IP address
TOPIC = "attendance/radar"
client = mqtt.Client()
client.connect(BROKER_IP, 1883, 60)

# --- Logic Variables ---
radar = RD03D()
radar.set_multi_mode(False)

streak_count = 0
last_direction = None

print("Sensor Node Started. Monitoring for streaks...")

while True:
    if radar.update():
        target1 = radar.get_target(1)
        #print('Distance: ', target1.distance, ' Angle: ', target1.angle
        if target1.distance < 600:
            current_direction = "ENTRY" if target1.angle < -1 else "EXIT"
            print('Distance: ', target1.distance, ' Angle: ', target1.angle, 'Sign: ', current_direction) 
            # Check if direction is the same as the last reading
            if current_direction == last_direction:
                streak_count += 1
            else:
                streak_count = 1  # Reset if direction changed
                last_direction = current_direction

            # Only trigger on exactly 3 in a row
            if streak_count == 3:
                #print(f"STREAK DETECTED: Sending {current_direction} to Inference Pi")
                client.publish(TOPIC, current_direction)
                streak_count = 0 # Reset after sending to wait for next 3
    
    else:
        # Reset streak if we lose tracking
        streak_count = 0
        last_direction = None

    time.sleep(0.5)