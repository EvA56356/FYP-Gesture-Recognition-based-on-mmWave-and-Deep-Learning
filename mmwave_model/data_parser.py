import pandas
import os
import keyboard
import time
import numpy as np
import tensorflow as tf
import asyncio
import tensorboard
from tensorflow import keras
from pymmWave.utils import load_cfg_file
from pymmWave.sensor import Sensor
from pymmWave.IWR6843AOP import IWR6843AOP
from asyncio import get_event_loop, sleep
print(tf.version.VERSION)
model = keras.models.load_model("mmwave_CNN.h5")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" Selects the mode of operation ,either live data capture or collection 
    will create the needed file structure and generate any needed csv files """
def user_mode():
    mode = 0
    file_index = 1
    dir_path = ''
    parent_dir = "./data/"
    sensor_data = pandas.DataFrame([], columns=["x", "y", "z", "Doppler"])
    print("Select Mode: ")
    while not mode or mode > 2:
        mode = int(input("[1] - Gesture Classification  | [2] - Data Capture \n"))
        print(mode)

    match mode:

        case (1):
            print("[1] is place holder for now")
            new_gesture_name = "place holder"
            dir_path = parent_dir

        case (2):
            print("[2] Data Capture Selected")
            new_gesture_name = str(input("Name of gesture: "))
            dir_path = os.path.join(parent_dir, new_gesture_name)

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
                sensor_data.to_csv(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv", index=False)
            else:
                print("Gesture Already Exists, will create new dataset within gesture file")
                while (os.path.exists(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv")):
                    file_index += 1

                sensor_data.to_csv(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv", index=False)

    return new_gesture_name, dir_path, mode, file_index

""" Sets up the connection ports for the mmwave sensor, initialises sensor,
    sends config file, etc. """
def configure_sensor():

    sensor1 = IWR6843AOP("1", verbose=False)
    file = load_cfg_file("mmwave_config/xwr68xx_AOP_config_10FPS_maxRange_30cm.cfg")

    # Your CONFIG serial port name
    config_connected = sensor1.connect_config('COM7', 115200)
    if not config_connected:
       print("Config connection failed.")
       exit()

    # Your DATA serial port name
    data_connected = sensor1.connect_data('COM8', 921600)
    if not data_connected:
       print("Data connection failed.")
       exit()

    if not sensor1.send_config(file, max_retries=1):
       print("Sending configuration failed")
       exit()

    # Sets up doppler filtering to remove static noise
    sensor1.configure_filtering(0.01)  # 0.05

    return sensor1

def VLC_Control(predicted_gesture, playing):

    match predicted_gesture:

        case("Forward Palm"): 

            if playing:

                keyboard.send("Ctrl+k")
                playing = 0

            elif not playing: 
                print("I am in forward palm not playing!")
                keyboard.send("Shift+l")
                playing = 1

        case("Left Swipe"):
                keyboard.send("left")

        case("Right Swipe"):
                keyboard.send("right")

        case("Up Swipe"):
                for _ in range(10):

                    keyboard.send("up")

        case("Down Swipe"):
                for _ in range(10):

                    keyboard.send("down")

""" Simple class to print data from the sensor, inherits from the default Sensor class
    in pymmwave """
async def print_data(sens: Sensor, gesture_name, directory, mode, index):

    cache_flush = 0
    playing = 1
    train_data = []
    live_test_data = []
    data_to_store = []
    encoded_labels = ["Left Swipe", "Right Swipe", "Up Swipe", "Down Swipe", "Forward Palm"]
    encoded_labels = np.array(encoded_labels)

    await asyncio.sleep(1)
    while sens.is_alive():

            sensor_object_data = await sens.get_data()  # get_data() -> DopplerPointCloud.
            incoming_data_array = sensor_object_data.get()
            print("\n New scan!")
            #print(incoming_data_array)
            #print(sens.get_update_freq())
            if cache_flush:
                    incoming_data_array = []
                    cache_flush = 0
                    print("\n Cache flushed!")

            if np.all(np.sum(incoming_data_array)):

                   if np.size(incoming_data_array) > 4:

                        average_of_data = np.sum(incoming_data_array, axis=0) / incoming_data_array.shape[0]
                        average_of_data = average_of_data.reshape(1, 4).astype("float32")
                        train_data.append(average_of_data)

                   else: train_data.append(incoming_data_array)

                   #print('\nTrain_data is: ', train_data)
                   print(np.size(train_data))

                   if mode and len(train_data) >= 3:
                        live_test_data = np.stack(train_data)
                        live_test_data = live_test_data.reshape(-1, 3, 4).astype("float32")
                        print('\n Live test data shape: ', live_test_data.shape)
                        prediction = encoded_labels[tf.argmax(model.predict(live_test_data), axis=1)]
                        confidence = np.max((100 * model.predict(live_test_data))) # Get the max confidence of the gesture
                        print(f"\nThe predicted gesture is: {prediction}, and the confidence is: {confidence}")
                        #VLC_Control(prediction, playing)
                        train_data = []
                        live_test_data = []
                        cache_flush = 1
                        time.sleep(2)

                   elif mode == 2 and np.size(train_data) > 2000: # 4000 for 1002 data points, 2000 for 502
                        data_to_store = np.vstack(train_data)
                        gesture_data = pandas.DataFrame(data_to_store)
                        gesture_data.to_csv(f"{directory}/{gesture_name}_{index}.csv", mode='a', index=False, header=False)
                        exit()

def main():

    gesture_name, directory, mode, index = user_mode()
    sensor = configure_sensor()

    event_loop = get_event_loop()
    event_loop.create_task(sensor.start_sensor())
    event_loop.create_task(print_data(sensor, gesture_name, directory, mode, index))
    event_loop.run_forever()

if __name__ == "__main__":
    main()