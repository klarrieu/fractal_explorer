# Fractal Explorer

![](example.png)

This repository contains a GUI for visual exploration of fractals.

Current capabilities include:

- adjustment of applied colormap
- navigation (up, down, left, right) and zooming
- saving images
- loading position/colormap from previously saved images

As navigation occurs, the fractal image is dynamically recomputed using the pyCUDA architecture.

Future improvements include:

- addition of more fractal systems
- option to adjust orbit trap used for coloring
- image-based orbit trapping

## Requirements

- Python 3
- an Nvidia GPU with pyCUDA installed (`conda install cudatoolkit=9.0`)