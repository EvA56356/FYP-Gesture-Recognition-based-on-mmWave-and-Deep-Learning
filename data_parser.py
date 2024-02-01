import pandas
import os
import keyboard
import time
import numpy as np
import tensorflow as tf
import asyncio
from tensorflow import keras
from pymmWave.utils import load_cfg_file
from pymmWave.sensor import Sensor
from pymmWave.IWR6843AOP import IWR6843AOP
from asyncio import get_event_loop, sleep
print(tf.version.VERSION)
model = keras.models.load_model("ConvNet.h5")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" Selects the mode of operation, either live data classification or collection 
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
            print("[1] Live Data Capture Selected")
            new_gesture_name = "place holder" #Stops errors
            dir_path = parent_dir

        case (2):
            print("[2] Data Capture Selected")
            new_gesture_name = str(input("Name of gesture: "))
            dir_path = os.path.join(parent_dir, new_gesture_name)
            #Check for excisting gesture name folder
            if not os.path.exists(dir_path):
                os.mkdir(dir_path) #Create it if needed
                sensor_data.to_csv(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv", index=False)
            else: #Else create a new csv file in the folder to store the data in
                print("Gesture Already Exists, will create new dataset within gesture file")
                while (os.path.exists(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv")):
                    file_index += 1

                sensor_data.to_csv(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv", index=False)

    return new_gesture_name, dir_path, mode, file_index

""" Sets up the connection ports for the mmwave sensor, initialises sensor,
    sends config file, etc. 
    
    Function taken from the example given in PymmWave tutorial          """
def configure_sensor():

    sensor1 = IWR6843AOP("1", verbose=False)
    file = load_cfg_file("mmwave_config/xwr68xx_AOP_config_10FPS_maxRange_30cm.cfg") #Select the config file to upload

    # Windows port names
    config_connected = sensor1.connect_config('COM13', 115200)
    if not config_connected:
       print("Config connection failed.")
       exit()

    # Your DATA serial port name
    data_connected = sensor1.connect_data('COM14', 921600)
    if not data_connected:
       print("Data connection failed.")
       exit()
    # WIll exit after a fixed number of failed attempts
    if not sensor1.send_config(file, max_retries=5):
       print("Sending configuration failed")
       exit()

    # Sets up doppler filtering to remove static noise
    sensor1.configure_filtering(0.01)

    return sensor1

""" Small self explanatory function to control the keyboard upon the detection of certain gestures
    Takes input from the print_data function in the form of a string """
def VLC_Control(predicted_gesture):

    match predicted_gesture:

        case("Forward Palm"): 
                keyboard.send("space")

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
            # Small check to remove the bug that causes extra data to fill the array.
            if cache_flush: # CAN CAUSE PERFORMANCE LOSS ON SLOW MACHINES
                    incoming_data_array = []
                    cache_flush = 0
                    print("\n Cache flushed!")

            if np.all(np.sum(incoming_data_array)): #If the sensor is returning data check

                   if np.size(incoming_data_array) > 4: # Average the data if multiple objects are found

                        average_of_data = np.sum(incoming_data_array, axis=0) / incoming_data_array.shape[0]
                        average_of_data = average_of_data.reshape(1, 4).astype("float32") # Correct the shape
                        train_data.append(average_of_data)

                   else: train_data.append(incoming_data_array)
                   print(np.size(train_data))
                   # If live classification mode was selected
                   if mode == 1 and len(train_data) >= 3: #When the amount of data is equal to the input size of the CNN
                        live_test_data = np.stack(train_data)
                        live_test_data = live_test_data.reshape(-1, 3, 4).astype("float32") #Stack a reshape to input
                        prediction = encoded_labels[tf.argmax(model.predict(live_test_data), axis=1)] # Predict a gesture
                        confidence = np.max((100 * model.predict(live_test_data))) # Get the max confidence of the gesture
                        print(f"\nThe predicted gesture is: {prediction}, and the confidence is: {confidence}")
                        VLC_Control(prediction) # Control the keyboard depending on the gesture
                        train_data = []
                        live_test_data = []
                        cache_flush = 1
                        time.sleep(2) # Allow the user time to move hand away

                   # If data collection mode is selected
                   # When array is the amount of data needed 
                   elif mode == 2 and np.size(train_data) > 2000: # 4000 for 1002 data points, 2000 for 502
                        data_to_store = np.vstack(train_data)
                        gesture_data = pandas.DataFrame(data_to_store) # Store the array data in csv file
                        gesture_data.to_csv(f"{directory}/{gesture_name}_{index}.csv", mode='a', index=False, header=False)
                        exit() # close program and shut sensor off.

def main():

    gesture_name, directory, mode, index = user_mode()
    sensor = configure_sensor()

    event_loop = get_event_loop()
    event_loop.create_task(sensor.start_sensor())
    event_loop.create_task(print_data(sensor, gesture_name, directory, mode, index))
    event_loop.run_forever()

if __name__ == "__main__":
    main()