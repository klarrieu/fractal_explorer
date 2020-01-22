from numba import cuda
from matplotlib.pyplot import show
import numpy as np
import math

image = None

def mandel(x, y, max_iters, trap='iterations'):
    """
    Determine if x + iy is in Mandelbrot set
    """
    c = x + 1j * y
    z = 0.0j
    if trap == 'iterations':
        for i in range(max_iters):
            z = z*z + c
            # check divergence criteria
            if (z.real * z.real + z.imag * z.imag) >= 4:
                return i
        # if didn't diverge
        return max_iters

    if trap == 'iters-smooth':
        for i in range(max_iters):
            z = z*z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                return i + (4 - abs(z)) / 2
        return max_iters

    if trap == 'smooth':
        for i in range(max_iters):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                return i + 1 + 1 / math.log(2.0) * math.log(math.log(2.0)/math.log(abs(z)))
        return max_iters

    if trap == 'magnitude':
        for i in range(max_iters):
            z = z*z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                return (z.real * z.real + z.imag * z.imag) ** 0.5
        return (z.real * z.real + z.imag * z.imag) ** 0.5


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
            image[y, x] = mandel_gpu(real, imag, iters, 'iterations')


def generate_img(centerX=-0.7, centerY=0, zoom=1, res=1080, iters=20, aspect=3 / 2):
    # generate fractal image using GPU

    # define input image array
    x_res = res
    y_res = int(res * aspect)
    gimage = np.zeros((x_res, y_res), dtype=np.uint64)

    # setup GPU array dims
    blockdim = (32, 8)
    griddim = (32, 16)

    # send gimage to GPU
    d_image = cuda.to_device(gimage)

    # scaling factor (0.8 for every integer increase in zoom)
    z_factor = (0.8 ** (zoom - 1))
    # half window sizes
    dx = 1 / 2 * 3 * z_factor
    dy = 1 / 2 * 2 * z_factor

    # compute on GPU
    mandel_kernel[griddim, blockdim](centerX - dx,
                                     centerX + dx,
                                     centerY - dy,
                                     centerY + dy,
                                     d_image,
                                     iters)
    # return to CPU
    d_image.to_host()

    return gimage


if __name__ == '__main__':
    img = generate_img()
    show(img)
