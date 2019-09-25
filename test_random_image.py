# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import cv2
import numpy as np
from numpy import vstack
# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

# load source image
src_image = load_image('test_set\sat.jpg')
gt_image = load_image('test_set\sat_gt.jpg')
#cv2.imshow('Input', src_image[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
print('Loaded', src_image.shape)
# load model
model = load_model('model_032880.h5')
# generate image from source
gen_image = model.predict(src_image)
# scale from [-1,1] to [0,1]
#--------------

#cv2.imshow('dst_rt', gen_image[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# plot source, generated and target images

#-----------------
#pyplot.imshow(src_image[0])
#pyplot.show()
#pyplot.imshow(gt_image[0])
#pyplot.show()
#gen_image = (gen_image + 1) / 2.0
#pyplot.imshow(gen_image[0])
#pyplot.axis('off')
#pyplot.show()


def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()
	
plot_images(src_image, gen_image, gt_image)






