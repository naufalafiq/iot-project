import paho.mqtt.client as paho
from paho import mqtt
import time
import json
import pymongo

# Define all the necessary values to connect to the HiveMQ broker
broker = "bc750697586e4ac7ab885ccb264207da.s1.eu.hivemq.cloud"
port = 8883
username = "Astainterns"
password = "Qu8nSJ4Tesdbp"
topic = "TELEMETRIX/TRUSTED/ASTA/Interns/#"

url = "<MongoDB Database>"
myclient = pymongo.MongoClient(url)
database = myclient["asta"]
collection = database["telemetrix"]

max_reconnect = 5
max_reconnect_delay = 5

# Callback to be called when the broker responds to a subscribe request to a topic
def on_subscribe(client, userdata, mid, granted_qos):
    a = 1
    # Callback for subscribing to 1 topic
    print(f"Subscribed to topic '{topic}' with QoS {granted_qos}") 
    # Callback for subscribing to multiple topics
    #for i in range(len(topic)):
            #print(f"Subscribed to topic '{topic[i][0]}' with QoS {granted_qos[i]}")

# Callback to be called when the broker reponds to our connection request
def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"Connected with result code {reason_code}")
    else:
        print(f"Connection failed with code {reason_code}")

# Callback to be called when a message has been received on a topic that the client subscribes to
def on_message(client, userdata, message):
    msg = message.payload
    decoded = msg.decode()
    print()
    print(f"Received message '{msg.decode()}' from '{message.topic}'")
    payload = json.loads(decoded)
    collection.insert_one(payload)
    print(f"The message has been inserted successfully.")
    
# Callback to be called when the client disconnects from the broker
def on_disconnect(client, userdata, flags, reason_code):
    reconnect_count = 0
    print(f"Disconnected with result code {reason_code}")
    while reconnect_count < max_reconnect:
        print(f"Reconnecting in {max_reconnect_delay} seconds...")
        time.sleep(max_reconnect_delay)
        try:
            client.reconnect()
            print("Reconnected successfully")
        except Exception:
            print(f"Exception Error. Reconnect failed. Retrying...")
        
        reconnect_count += 1
    print(f"Reconnect failed after {reconnect_count} attempts. Exiting...")


print("Connecting...")
# The main class for use communicating with an MQTT broker
client = paho.Client()
# Sets on_connect to the previous callback defined
client.on_connect = on_connect

# tls_set configures the network encryption and authentication options
client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
# Sets a username and optionally a password for broker authentication
client.username_pw_set(username, password)

# Connects to the MQTT broker which establishes the underlying connection and transmits a CONNECT packet
# Tries to connect unless it takes too long and terminates the program
client.connect(broker, port)

# Sets the on_message and on_subscribe to the previous callbacks defined
client.on_message = on_message
client.on_subscribe = on_subscribe
client.on_disconnect = on_disconnect

# Subscribes the client to one or more topics
client.subscribe(topic, qos=1)

# This calls the network loop the network loop functions for you in an infinite blocking loop
client.loop_forever()
