import time

import gpiozero

from ..utils import log_info


def __wait_and_squirt(should_stop, predictions, labels):
    output = gpiozero.DigitalOutputDevice('GPIO14')

    while not should_stop.is_set():
        class_predictions = dict(zip(labels, predictions))
        bad_cat_detected = class_predictions['black-cat'] >= 0.9

        if bad_cat_detected:
            log_info('Squirting cat')
            output.on()
            time.sleep(2.0)
            output.off()
        
        should_stop.wait(timeout=0.1)


def start_squirter(**kwargs):
    should_stop = kwargs['should_stop']
    try:
        __wait_and_squirt(**kwargs)
    except KeyboardInterrupt:
        should_stop.set()
