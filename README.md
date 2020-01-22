# Fractal Explorer

![](example.png)

This repository contains a GUI for visual exploration of fractals.

Current capabilities include:

- adjustment of applied colormap and hillshade
- navigation (up, down, left, right) and zooming
- saving images
- loading position/colormap from previously saved images
- saving video: zooms into current position

As navigation occurs, the fractal image is dynamically recomputed using the CUDA architecture.

Future improvements include:

- addition of more fractal types
- option to adjust orbit trap function
- image-based orbit trapping

## Requirements

- Python 3.x
- an Nvidia GPU with CUDA Python installed. 
- *optional*: OpenCV-Python (for saving videos)

## Installation and Usage
1. Ensure your computer is equipped with an Nvidia GPU.
2. Install [CUDA](https://developer.nvidia.com/cuda-toolkit).
3. Install [Python CUDA packages](https://developer.nvidia.com/how-to-cuda-python) (in a conda environment: `conda install numba cudatoolkit=9.0 pyculib`).
4. Install [OpenCV-Python](https://pypi.org/project/opencv-python/): `pip install opencv-python`  *(for saving videos)*
5. Run the `fractal_explorer.py` script in this repository and have fun!

Images created using the `Save image` button are saved to an `\images\` directory in the local repository. Upon saving, the output filename is also printed to the console. 

Loading a previously saved image with the `Load image` button will reset the window location, zoom, and colormap according to the loaded image.

Videos can also be saved using the `Save video` showing a gradual zoom to the current location.
