import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import fractal_cuda as fcuda


class FractalExplorer(object):
    def __init__(self):

        self.root = tk.Tk()
        self.root.title('Fractal Explorer')

        self.cwd = os.path.dirname(__file__)
        self.out_dir = os.path.join(self.cwd, 'images\\')
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.frac_types = ['Mandelbrot']
        self.cmaps = list(plt.cm.datad.keys())

        # setup canvas
        self.fig = Figure()
        self.fig1 = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.fig1.set_axis_off()
        self.fig.add_axes(self.fig1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        self.options_frame = tk.Frame(self.root)
        self.options_frame.pack(side=tk.LEFT, padx=20)
        # select fractal type
        self.l_frac_type = tk.Label(self.options_frame, text='fractal type:')
        self.l_frac_type.grid(sticky=tk.E, row=1, column=0)
        self.c_frac_type = ttk.Combobox(self.options_frame, values=self.frac_types, state='readonly')
        self.c_frac_type.current(0)
        self.c_frac_type.grid(sticky=tk.W, row=1, column=1)

        # select colormap
        self.l_cmap = tk.Label(self.options_frame, text='color map:')
        self.l_cmap.grid(sticky=tk.E, row=2, column=0)
        self.c_cmap = ttk.Combobox(self.options_frame, values=self.cmaps, state='readonly')
        self.c_cmap.current(self.cmaps.index('cubehelix'))
        self.c_cmap.grid(sticky=tk.W, row=2, column=1)
        self.c_cmap.bind('<<ComboboxSelected>>', self.update_cmap)

        # select orbit trap
        self.l_otrap = tk.Label(self.options_frame, text='orbit trap:')
        self.l_otrap.grid(sticky=tk.E, row=3, column=0)
        self.c_otrap = ttk.Combobox(self.options_frame, values=['magnitude'], state='readonly')
        self.c_otrap.current(0)
        self.c_otrap.grid(sticky=tk.W, row=3, column=1)

        # zoom level
        self.l_zoom = tk.Label(self.options_frame, text='zoom:')
        self.l_zoom.grid(sticky=tk.E, row=4, column=0)
        self.zoom_var = tk.StringVar()
        self.zoom_var.set('1')
        self.s_zoom = tk.Spinbox(self.options_frame, from_=1, to=1e9, textvariable=self.zoom_var, width=20, command=self.update_zoom)
        self.s_zoom.grid(sticky=tk.W, row=4, column=1)

        # navigation buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, padx=20)
        w, h = 2, 1
        self.b_up = tk.Button(self.control_frame, text='↑', command=self.update_up, width=w, height=h)
        self.b_up.grid(row=2, column=5)
        self.b_down = tk.Button(self.control_frame, text='↓', command=self.update_down, width=w, height=h)
        self.b_down.grid(row=4, column=5)
        self.b_left = tk.Button(self.control_frame, text='←', command=self.update_left, width=w, height=h)
        self.b_left.grid(row=3, column=4)
        self.b_right = tk.Button(self.control_frame, text='→', command=self.update_right, width=w, height=h)
        self.b_right.grid(row=3, column=6)

        # save button
        self.b_save = tk.Button(self.options_frame, text='Save image', command=self.save_img, width=10)
        self.b_save.grid(row=8, column=0, pady=10)

        # load position from image
        self.b_load = tk.Button(self.options_frame, text='Load', command=self.load_img, width=10)
        self.b_load.grid(row=8, column=1, pady=10)

        # initialize image
        self.fractal_type = self.c_frac_type.get()
        self.cmap = self.c_cmap.get()
        self.zoom = float(self.s_zoom.get())
        self.update_iters()
        self.centerX = -0.7
        self.centerY = 0
        self.update_image()

        # launch GUI
        self.root.mainloop()

    def update_fractal_type(self):
        print('under construction')

    def update_orbit_trap(self):
        print('under construction')

    def update_image(self):
        # updates image when navigation changes
        self.gimage = fcuda.generate_img(centerX=self.centerX, centerY=self.centerY, zoom=self.zoom, iters=20*self.zoom)
        self.fig1.cla()
        self.img = self.fig1.imshow(self.gimage, cmap=self.cmap)
        self.fig1.axis('off')
        self.canvas.draw()

    def update_cmap(self, event):
        # update colormap without recomputing image
        self.cmap = self.c_cmap.get()
        self.img.set_cmap(self.cmap)
        self.canvas.draw()

    def update_zoom(self):
        # update zoom level
        self.zoom = float(self.s_zoom.get())
        self.update_iters()
        self.update_image()

    def update_iters(self):
        self.z_factor = (0.8 ** (self.zoom - 1))
        self.iters = int(2 / self.z_factor)

    def update_left(self):
        # update when left button pressed
        self.centerX -= 0.1 * self.z_factor
        self.update_image()

    def update_right(self):
        self.centerX += 0.1 * self.z_factor
        self.update_image()

    def update_up(self):
        self.centerY -= 0.1 * self.z_factor
        self.update_image()

    def update_down(self):
        self.centerY += 0.1 * self.z_factor
        self.update_image()

    def save_img(self):
        img_name = '%s_%s_%s_%s_%s.png' % (self.c_frac_type.get(),
                                           str(self.centerX).replace('.', 'pt'),
                                           str(self.centerY).replace('.', 'pt'),
                                           str(self.zoom).replace('.', 'pt'),
                                           self.cmap)
        img_path = os.path.join(self.out_dir, img_name)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(self.gimage, cmap=self.cmap)
        plt.axis('off')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(img_path,
                    bbox_inches=extent,
                    pad_inches=0,
                    transparent=True,
                    dpi=600)
        plt.cla()
        print('Saved image: %s' % img_path)

    def load_img(self):
        f = askopenfilename(parent=self.root,
                            initialdir=self.out_dir,
                            title='Choose a previously saved image:',
                            filetypes=[['PNG', '.png']])
        name = os.path.basename(f)
        vals = name.replace('pt', '.').replace('.png', '').split('_')

        self.fractal_type = vals[0]
        self.centerX = float(vals[1])
        self.centerY = float(vals[2])
        self.zoom = float(vals[3])
        self.cmap = vals[4]
        if len(vals) > 5:
            self.cmap = self.cmap + '_r'

        self.c_frac_type.set(self.fractal_type)
        self.zoom_var.set(self.zoom)
        self.c_cmap.current(self.cmaps.index(self.cmap))

        self.update_image()


if __name__ == '__main__':
    FractalExplorer()
