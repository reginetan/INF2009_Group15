import serial
import time
import math

class Target:
    def __init__(self, x, y, speed, pixel_distance):
        self.x = x                  # mm
        self.y = y                  # mm
        self.speed = speed          # cm/s
        self.pixel_distance = pixel_distance  # mm
        self.distance = math.sqrt(x**2 + y**2)
        self.angle = math.degrees(math.atan2(x, y))
    
    def __str__(self):
        return ('Target(x={}mm, y={}mm, speed={}cm/s, pixel_dist={}mm, '
                'distance={:.1f}mm, angle={:.1f}°)').format(
                self.x, self.y, self.speed, self.pixel_distance, self.distance, self.angle)

class RD03D:
    SINGLE_TARGET_CMD = bytes([0xFD, 0xFC, 0xFB, 0xFA, 0x02, 0x00, 0x80, 0x00, 0x04, 0x03, 0x02, 0x01])
    MULTI_TARGET_CMD  = bytes([0xFD, 0xFC, 0xFB, 0xFA, 0x02, 0x00, 0x90, 0x00, 0x04, 0x03, 0x02, 0x01])
    
    def __init__(self, uart_port='/dev/ttyAMA0', baudrate=256000, multi_mode=True):
        self.uart = serial.Serial(uart_port, baudrate, timeout=0.1)
        self.targets = []  # Stores up to 3 targets
        self.buffer = b''  # Buffer to handle split messages
        time.sleep(0.2)
        self.set_multi_mode(multi_mode)
    
    def set_multi_mode(self, multi_mode=True):
        """Set Radar mode: True=Multi-target, False=Single-target"""
        cmd = self.MULTI_TARGET_CMD if multi_mode else self.SINGLE_TARGET_CMD
        self.uart.write(cmd)
        self.uart.flush()  # Force immediate send
        time.sleep(0.2)
        self.uart.reset_input_buffer()  # Clear buffer after switching
        self.buffer = b''  # Clear internal buffer too
        self.multi_mode = multi_mode
    
    @staticmethod
    def parse_signed16(high, low):
        raw = (high << 8) + low
        sign = 1 if (raw & 0x8000) else -1
        value = raw & 0x7FFF
        return sign * value
    
    def _decode_frame(self, data):
        targets = []
        if len(data) < 30 or data[0] != 0xAA or data[1] != 0xFF or data[-2] != 0x55 or data[-1] != 0xCC:
            return targets  # invalid frame
        
        for i in range(3):
            base = 4 + i*8
            x = self.parse_signed16(data[base+1], data[base])
            y = self.parse_signed16(data[base+3], data[base+2])
            speed = self.parse_signed16(data[base+5], data[base+4])
            pixel_dist = data[base+6] + (data[base+7] << 8)
            targets.append(Target(x, y, speed, pixel_dist))
        
        return targets
    
    def _find_complete_frame(self, data):
        """Find a complete frame in the data buffer"""
        # Look for frame start (0xAA 0xFF)
        start_idx = -1
        for i in range(len(data) - 1):
            if data[i] == 0xAA and data[i+1] == 0xFF:
                start_idx = i
                break
        
        if start_idx == -1:
            return None, data  # No frame start found, keep all data
        
        # Look for frame end (0x55 0xCC) after the start
        for i in range(start_idx + 2, len(data) - 1):
            if data[i] == 0x55 and data[i+1] == 0xCC:
                # Found complete frame
                frame = data[start_idx:i+2]
                remaining = data[i+2:]
                return frame, remaining
        
        # Frame start found but no end yet, keep data from start
        return None, data[start_idx:]
    
    def update(self):
        """Update internal targets list with latest data from radar."""
        # Read all available data and add to buffer
        if self.uart.in_waiting > 0:
            new_data = self.uart.read(self.uart.in_waiting)
            self.buffer += new_data
        
        # If buffer gets too large, keep only the most recent data
        if len(self.buffer) > 300:  # ~10 frames worth
            self.buffer = self.buffer[-150:]  # Keep last ~5 frames
        
        # Try to extract the MOST RECENT complete frame
        latest_frame = None
        temp_buffer = self.buffer
        
        while True:
            frame, temp_buffer = self._find_complete_frame(temp_buffer)
            if frame:
                latest_frame = frame  # Keep updating to get the latest
            else:
                break
        
        # Update buffer to remaining data after last complete frame
        if latest_frame:
            # Find where the latest frame ends in the original buffer
            frame_end_pos = self.buffer.rfind(latest_frame) + len(latest_frame)
            self.buffer = self.buffer[frame_end_pos:]
            
            decoded = self._decode_frame(latest_frame)
            if decoded:
                self.targets = decoded
                return True  # Successful update
        
        return False  # No valid frame found
    
    def get_target(self, target_number=1):
        """Get a target by number (1-based index)."""
        if 1 <= target_number <= len(self.targets):
            return self.targets[target_number - 1]
        return None  # No such target
    
    def close(self):
        """Close the UART connection"""
        if self.uart.is_open:
            self.uart.close()