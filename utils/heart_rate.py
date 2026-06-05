def calculate_heart_rate(blink_count):

    bpm = 60 + (blink_count * 2)

    if bpm > 100:
        bpm = 100

    return bpm