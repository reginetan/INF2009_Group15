# Hardware setup pins
# mmwave tx to pi rx
# mmwave rx to pi tx
# mmwave gnd to pi gnd
# mmwave 5v to pi 5v
from rd03d import RD03D
import time
# from t2_pipeline import run_t2_pipeline

# Initialize radar with Pi 5 UART settings
radar = RD03D()  # Uses /dev/ttyAMA0 by default

radar.set_multi_mode(False)   # Switch to multi-target mode set to True

while True:
    if radar.update():
        target1 = radar.get_target(1)
        #target2 = radar.get_target(2)
        #target3 = radar.get_target(3)
        if target1.distance < 600:
            direction = "ENTRY" if target1.angle < -1 else "EXIT"
            print("Human Presence Detected. Purpose: " + direction)
            
            #result = run_t2_pipeline(direction)
            #if result:
                #print(f"[Radar] T2 pipeline completed | direction={result['direction']} | conf={result['confidence']:.2f}")
        
        #print('1 dist:', target1.distance, 'mm Angle:', target1.angle, " deg Speed:", target1.speed, "cm/s X:", target1.x, "mm Y:", target1.y, "mm")
        #print('2 dist:', target2.distance, 'mm Angle:', target2.angle, " deg Speed:", target2.speed, "cm/s X:", target2.x, "mm Y:", target2.y, "mm")
        #print('3 dist:', target3.distance, 'mm Angle:', target3.angle, " deg Speed:", target3.speed, "cm/s X:", target3.x, "mm Y:", target3.y, "mm \n")
    
    else:
        print('No radar data received.')
    
    time.sleep(0.2)