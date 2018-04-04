import skimage as sk
import skimage.io as skio

img = skio.imread("./images/brown-falcon.jpg")

print(img.size)