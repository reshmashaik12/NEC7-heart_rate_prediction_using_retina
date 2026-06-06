import random
import time

current_bpm = 72
previous_blink_count = 0
last_update = time.time()

def calculate_heart_rate(blink_count):

    global current_bpm
    global previous_blink_count
    global last_update

    current_time = time.time()

    blink_difference = blink_count - previous_blink_count

    # Increase BPM when blinking
    if blink_difference > 0:

        current_bpm += random.uniform(1, 3)

    else:

        # Natural decrease
        current_bpm -= random.uniform(0.1, 0.4)

    # Small natural variation
    current_bpm += random.uniform(-0.5, 0.5)

    # Human range
    current_bpm = max(68, min(current_bpm, 105))

    previous_blink_count = blink_count
    last_update = current_time

    return int(current_bpm)