import logging, time

import paho.mqtt.publish as publish


def notify(mqtt_host, mqtt_port, mqtt_user, mqtt_pass, mqtt_topic, should_publish, should_stop):
    while not should_stop.is_set():
        if should_publish.is_set():
            logging.info('Publishing to MQTT')
            
            publish.single(
                mqtt_topic,
                hostname=mqtt_host,
                port=mqtt_port,
                auth={
                    'username': mqtt_user,
                    'password': mqtt_pass,
                }
            )

            should_publish.clear()
        
        time.sleep(0.25)
