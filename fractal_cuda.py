from numba import cuda
from numba import *
from matplotlib.pyplot import imshow, show
import numpy as np

image = None


def mandel(x, y, max_iters):
    """
    Determine if x + iy is in Mandelbrot set
    """
    c = x + 1j * y
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        # check divergence criteria
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    # if didn't diverge
    return max_iters


# compile mandel to run on GPU
mandel_gpu = cuda.jit(device=True)(mandel)


# define mandel kernel which applies mandel_gpu to GPU array
@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel_gpu(real, imag, iters)


def generate_img(centerX=-0.5, centerY=0, zoom=1, res=1080, iters=20, aspect=3 / 2):
    # generate fractal image using GPU

    # define input image array
    x_res = res
    y_res = int(res * aspect)
    gimage = np.zeros((x_res, y_res), dtype=np.uint64)
    # setup GPU array dims
    blockdim = (32, 8)
    griddim = (32, 16)

    d_image = cuda.to_device(gimage)

    dx = 1 / 2 * 3 / (zoom ** 2)
    dy = 1 / 2 * 2 / (zoom ** 2)

    mandel_kernel[griddim, blockdim](centerX - dx, centerX + dx, centerY - dy, centerY + dy, d_image, iters)
    d_image.to_host()

    # image = imshow(gimage, cmap='gist_ncar')
    return gimage


if __name__ == '__main__':
    img = generate_img()
    show(img)
