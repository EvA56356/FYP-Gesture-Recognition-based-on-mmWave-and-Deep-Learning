# FYP-Gesture-Recognition-based-on-mmWave-and-Deep-Learning
This project aims to develop a hand gesture system to control specific applications
using the Texas Instruments (TI) IWR6843AOPEVM sensor, an integrated
single-chip based on the frequency-modulated continuous wave (FMCW) technology,
and deep learning. The application of millimetre-wave (mmWave) can
provide the convenience of contactless control and compensate for the shortcomings
of privacy concerns compared to camera-based recognition systems. Each
hand gesture consists of three frames with a three-dimensional coordinate and radial
velocity (Doppler) parameter. The system can use the pymmWave library to
fetch 12 parameters for each gesture, which is pre-calculated from the sensor and
stored in a .csv file. The system stores the five hand gestures, each with a size
of 500, in separate files, labelled with their respective gestures as the filename.
The recorded training datasets are used to train a convolutional neural network
(CNN), achieving a 95.2% accuracy rate for five different non-static hand gestures:
right swipe, left swipe, up swipe, down swipe, and forward palm. The system can
use these five gestures to control various applications like music, video players, and
web browsers. Data visualization for the five training datasets is also conducted
to provide an intuition about the correctness of dataset collection. The overall
system shows high-quality applications and scalability regarding control functions
and system development. The significant shortcomings of this system are the need
for more support for recognizing complex gestures and static hand gestures, which
are significant areas for improvement in future work.
