import tensorflow as tf
import os
from util import * 
from PIL import Image
import scipy

bs = load_bsCoeff('./data/bsCoeff/1_bsCoeff.mat')
mat = scipy.io.loadmat('./data/result_8_0.1/2.mat')
curCP = np.array(mat['dstCP'])
curCP = np.transpose(curCP)
curCP = np.reshape(curCP, [9*9*9,3])
# fig = Figure()
# canvas = FigureCanvas(fig)
# ax = Axes3D(fig)
# ax.scatter(curCP[:,0],curCP[:,0],curCP[:,0],c='r')
# canvas.draw()
# img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
# width, height = (fig.get_size_inches() * fig.get_dpi())
# img = np.reshape(img, (int(height),int(width),3))

img = vis_image(bs, [curCP], 0, vis_path=None)[0]
im = Image.fromarray(img)
im.show()