# Helmet Detection and Traffic Signal Control System

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Overview 
:eyes:

This project leverages computer vision and machine learning to enhance road safety by automating helmet detection among motorcyclists and intelligently controlling traffic signals. The system uses a machine learning-based algorithm to detect motorcycle riders and determine if they are wearing a helmet. If a rider is detected without a helmet, their image is displayed. Additionally, a traffic signal control mechanism is implemented to prevent the signal from turning green until all riders in the frame are wearing helmets.

## Key Features
✨

* **Automatic Helmet Detection:** Utilizes computer vision to identify helmet usage in real-time.
* **Traffic Signal Control:** Dynamically adjusts traffic signals based on helmet detection to ensure compliance.
* **Real-time Object Detection:** Employs the YOLO model for fast and accurate object detection.

## Tech Stack
💻

This project utilizes the following technologies:

* **Programming Language:**
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
* **Deep Learning Framework:**
    <img src="https://img.shields.io/badge/TensorFlow-F9A01B?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
    or
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
* **Computer Vision Library:**
    <img src="https://img.shields.io/badge/OpenCV-272727?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
* **Object Detection Model:**
    <img src="https://img.shields.io/badge/YOLO-F7DF1E?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLO">
* **Data Processing Tools:**
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
##   Project Structure 📂

A typical project structure might look like this:
helmet-detection/  
├── data/  
│   ├── images/               # Directory for image datasets  
│   ├── videos/               # Directory for video datasets  
│   └── annotations/          # Annotation files (if applicable)  
├── models/  
│   ├── yolo_weights/         # YOLO model weights and config files  
│   └── saved_model/          # Saved TensorFlow/PyTorch models (if applicable)  
├── src/  
│   ├── main.py               # Main script to run the system  
│   ├── detection.py          # Helmet detection logic  
│   ├── signal_control.py     # Traffic signal control logic (if applicable)  
│   ├── utils.py              # Utility functions  
├── requirements.txt          # Python dependencies  
├── README.md                 # Project documentation  
└── ...                       # Other project-specific files  



## System Architecture
:building_construction:

The system integrates OpenCV for capturing video or visual data (e.g., from CCTV cameras) and the YOLO model for real-time object detection. Deep learning frameworks like TensorFlow or PyTorch are used to generate the weights file for the model.

## Problem Statement
🤔

The project addresses the critical issue of road safety by focusing on the detection of riders without helmets, which is a significant concern due to the high usage of motorcycles and the resulting increase in road accidents.

The project's main aim is to detect bike-riders without helmets using computer vision. If the person who is riding the motor-cycle is detected without helmet, then the picture of that rider is displayed on the screen, making him to feel embarrassed. A traffic signal control mechanism transitioning the signal to red, if any of the person is detected without a helmet.

## Proposed System
💡

Our proposed system utilizes deep learning frame works like Tensor flow or pytorch to generate the weights file, which is combined with the input datasets, used for training the model.

In this system we mainly use OPEN VISION COMPUTER SOURCE (OPEN CV) and YOU ONLY LOOK ONCE (YOLO) model for detection of objects(helmets) . Open CV is used for capturing the video or visual data through CCTV, cameras, etc. And YOLO model is used for real - time object detection and known for its speed and accuracy.

It divides the input data into grids and identifies the object, and detects whether the helmet is present or not.

Additionally, the traffic control mechanism transitions the signal into red when there is no detection or presence of helmet. And works like a regular signal when there is detection or presence of helmet.

## Installation Guide
:gear:

This guide provides a comprehensive walkthrough of installing and setting up the Helmet Detection and Traffic Signal Control System.

### 1\. Prerequisites
✅

Before proceeding with the installation, ensure that your system meets the following requirements:

* **Operating System:** Linux, macOS, or Windows
* **Python:** Python 3.6 or later is highly recommended. You can check your Python version by opening a terminal or command prompt and running:

    ```bash
    python --version
    ```

* **pip:** Python's package installer. It usually comes with Python. Verify it's installed:

    ```bash
    pip --version
    ```

* **Git:** For cloning the repository. You can download it from \[https://git-scm.com/\](https://git-scm.com/)

### 2\. Installation Steps
⬇️

1.  **Clone the Repository:**

    * Open a terminal or command prompt.
    * Navigate to the directory where you want to store the project.
    * Clone the repository using Git:

        ```bash
        git clone <repository_url> # Replace <repository_url> with the actual URL
        cd <repository_name> # Navigate into the cloned directory
        ```

2.  **Create a Virtual Environment (Recommended):**

    * It's best practice to create a virtual environment to isolate project dependencies.
    * **For Linux/macOS:**

        ```bash
        python3 -m venv venv # Create a virtual environment named "venv"
        source venv/bin/activate # Activate the environment
        ```

    * **For Windows:**

        ```bash
        python -m venv venv # Create a virtual environment named "venv"
        venv\\Scripts\\activate.bat # Activate the environment
        ```

3.  **Install Dependencies:**

    * Navigate to the project directory if you're not already there.
    * Install the required Python packages using pip:

        ```bash
        pip install -r requirements.txt
        ```

        *(**Important:** You must have a `requirements.txt` file in your project directory. This file lists all the necessary Python packages. Here's how to create it, if you haven't already: `pip freeze > requirements.txt` (Run this *after* installing the packages))*

4.  **Install TensorFlow or PyTorch:**

    * Choose either TensorFlow or PyTorch as your deep learning framework. *(The project description mentions either can be used)*
    * **TensorFlow (CPU):**

        ```bash
        pip install tensorflow
        ```

    * **TensorFlow (GPU - If you have a compatible NVIDIA GPU):**

        ```bash
        pip install tensorflow\[gpu]
        ```

        * You might need to install CUDA and cuDNN separately for GPU support. Refer to the TensorFlow documentation.

    * **PyTorch:**

        * Visit the PyTorch website (\[https://pytorch.org/get-started/locally/\](https://pytorch.org/get-started/locally/)) to get the specific installation command for your operating system, Python version, and CUDA version (if applicable). It will likely look something like:

            ```bash
            pip install torch torchvision torchaudio
            ```

5.  **Download YOLO Model Files:**
    :warning:
    * You'll need the YOLO model weights and configuration files.
    * ***(Since the source doesn't provide specifics on where to download these, you'll need to add instructions here based on the YOLO version you're using (e.g., YOLOv5, YOLOv8). Provide the download links and where to place the files in the project directory.)***
    * **Example (Conceptual - Replace with actual instructions):**
        * "Download the YOLOv5 weights file `yolov5s.pt` from \[link to YOLOv5 release].
        * Create a directory named `yolo_weights` in the project root.
        * Place the downloaded `yolov5s.pt` file in the `yolo_weights` directory."
        * "Download the YOLOv5 configuration file `yolov5s.yaml` from \[link to YOLOv5 release].
        * Place the downloaded `yolov5s.yaml` file in the `yolo_weights` directory."

### 3\. Configuration
:wrench:

1.  **Camera Input:**

    * In your main Python script (e.g., `main.py`), you'll need to configure how the system accesses the camera.
    * If using a webcam, you'll likely use a camera index (usually 0 for the default camera).
    * If using a video file, you'll provide the file path.
    * *(Provide code snippets or examples from your `main.py` to illustrate this)*
    * **Example:**

        ```python
        import cv2

        # To use the default webcam (index 0):
        cap = cv2.VideoCapture(0)

        # To use a video file:
        # cap = cv2.VideoCapture('path/to/your/video.mp4')
        ```

2.  **YOLO Model Paths:**

    * Ensure that the paths to the YOLO weights and configuration files are correctly specified in your Python script.
    * *(Provide code snippets from your project)*
    * **Example:**

        ```python
        YOLO_WEIGHTS = "yolo_weights/yolov5s.pt"
        YOLO_CONFIG = "yolo_weights/yolov5s.yaml"
        ```

3.  **Traffic Signal Control (If Applicable):**
    🚦
    * If your project includes controlling physical traffic signals, provide detailed instructions on:
        * The hardware interface used (e.g., serial communication, GPIO pins).
        * Any required libraries or drivers.
        * The communication protocol.
        * Configuration settings in your Python script.

### 4\. Running the System
▶️

1.  Open a terminal or command prompt.
2.  Navigate to the project directory.
3.  Activate the virtual environment (if you created one).
4.  Run the main Python script:

    ```bash
    python main.py
    ```

5.  The system should now start processing video input, detecting helmets, and (if configured) controlling traffic signals.

### 5\. Troubleshooting
:question:

* **Import Errors:** If you get "ImportError" messages, double-check that you've installed all the required packages in the correct environment.
* **Camera Issues:**
    * Make sure your camera is properly connected and working.
    * Check the camera index in your script.
    * Ensure that no other applications are using the camera.
* **YOLO Errors:**
    * Verify that the YOLO weights and configuration file paths are correct.
    * Make sure you've downloaded the correct YOLO files for the version you're using.
* **Performance:**
    * If the system is running slowly, consider using a GPU if possible.
    * You might need to adjust the YOLO model size or other parameters to optimize performance.

## Future Enhancements
:rocket:

This project has the potential for several enhancements to further improve its functionality and impact:

* **Expanded Object Detection:**
    * Include the detection of other traffic violations, such as detecting vehicles running red lights, illegal turns, or pedestrian violations.
    * Extend object detection to identify different types of vehicles (e.g., cars, trucks, buses) to gather more comprehensive traffic data.
* **Enhanced Traffic Management:**
    * Implement adaptive traffic signal control based on real-time traffic density and vehicle types to optimize traffic flow.
    * Integrate with navigation systems to provide real-time traffic updates and alternative route suggestions.
* **Improved Accuracy and Robustness:**
    * Explore advanced machine learning models or techniques to increase the accuracy and robustness of helmet detection, especially in varying lighting and weather conditions.
    * Incorporate methods to handle occlusions (partially hidden helmets) and different helmet types.
* **Data Analytics and Reporting:**
    * Store and analyze traffic violation data to identify high-risk areas and inform traffic management strategies.
    * Generate reports and visualizations on traffic patterns, helmet usage compliance, and violation statistics.
* **Integration with Law Enforcement:**
    * Enable real-time alerts to law enforcement agencies for immediate response to traffic violations.
    * Provide evidence (images/videos) of violations to aid in enforcement.
* **Mobile Application Development:**
    * Develop a mobile application for users to report traffic violations or access real-time traffic information.

These enhancements would contribute to a more comprehensive and effective intelligent traffic management system, further improving road safety and traffic efficiency.
