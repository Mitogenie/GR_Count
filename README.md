# GR Count

[![N|Solid](https://raw.githubusercontent.com/Mitogenie/GR_Count/master/misc/logo.png)](https://mic.med.virginia.edu/zong/)

### A tool to analyze mutant cell expansion kinetics.

![Build Status](https://raw.githubusercontent.com/Mitogenie/GR_Count/master/misc/ver.png)

##### User Interface
GR Count offers interactive, semi graphical user interface through Jupyter notebooks.
##### Software Requirements
Follow these instructions if you do not already have Anaconda or Jupyter notebooks installed on your computer.
</br>
  - GR Count is developed in Jupyter notebooks using Python 3. Since it benefits from a wide range of Python libraries, along with Jupyter notebooks, we suggest installing Anaconda distribution of Python (V 3.7). --> [Download Anaconda](https://www.anaconda.com/distribution/)
You may follow the rest of the instructions if you do not already have OpenCV installed on your computer.
  - While Anaconda installation takes care of most of the library requirements for CeMiA, there is only one more libary (OpenCV) that needs to be installed, which can be achieved through the command line. (You just need to copy and run the following command. (without the $))
    - Windows: Use Anaconda Prompt.
    - MacOS, Linux: Use Terminal.
```sh
$ pip install opencv-python==3.4.2.17
```
  - It is important to install this specific version of OpenCV for compatibility.
  - GR Count depends on grc.py to run. This module includes all the functions used in the develepment of GR Count. This file should be in the same folder as the GR_Count.ipynb jupyter notebook you are running.

### Where to start your analysis?
##### 1) Download the files on your computer
Once you have satistfied the requirements, you can start your analysis by downloading or cloning this repository on your computer. The simplest way is to download the whole directory by pressing the green button (top right) and download the ZIP file.
##### 2) Run the GR_Count file using Jupyter notebook 
##### 3) Follow the step-by-step instructions in the notebook to analyze your data

#### What to know before use
- Input: Individual sequence of images exported from your stack of images using ImageJ
  - Images should have the following naming format: name_of_the_image_z***_c***.tif
    - GR Count reads the stack and channel info ((z*** and c***) from each image to automate the measurements. So this formatting is required for the code to run.
      - Example: RGB_z011_c001: is the channel1 (red) from stack 11.
  - Number channels with stains: 2
      - Red and Green channels should be stained
- Output: Two csv files containing signal measurements over the stack of images, and the summary of the measurements.
  - Stack_measurements.csv: Includes signal measurements over the stack of images.
  - Stack_summary.csv: Includes statistical summary of the measured features across the stack. 

###### Instructions
- Run the GR_Count.ipynb using Jupyter notebook.
    - You may find this video on Jupyter notebooks very helpful: [Watch Here](https://youtu.be/HW29067qVWk)
        - Note: We are not affiliated with the owner of the above video. We just found this video on Jupyter notebooks very helpful, There are plenty of great tutorials about this subject, and you may use any source your prefer.
- The Jupyter notebook file provides the users with step-by-step instructions to analyze their data.

###### Development
- 500+ images (3 stack images) were used to develop and test GR Count.

#### Nomenclature of Features (What does each measure feature mean?)
###### Red Area
- Total red signal at each layer of the stack
###### Grean Area
- Total green signal at each layer of the stack
###### Red/Green Ratio
- The ratio of the red signal to green signal at each layer of the stack
###### Green/Red Ratio
- The ratio of the green signal to red signal at each layer of the stack
###### Total Green Area along the Stack
- The sum of green area of all the layers of the stack
###### Total Red Area along the Stack
- The sum of red area of all the layers of the stack
###### Green/Red Ratio of the stack
- The the ratio of "Total Green Area" to "Total Red Area"
###### Red/Green Ratio of the stack
- The the ratio of "Total Red Area" to "Total Green Area"

#### List of the libraries we used for development (A/Z)
- copy
- cv2
- ipywidgets
- matplotlib.pyplot
- numpy
- os
- pandas
- re
- seaborn
- skimage

