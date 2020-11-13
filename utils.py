import scipy.misc
import numpy as np
import copy
import os
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size, fine_size*2])
    img = img/127.5 - 1
    return img

def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def scale_shorter(pil_img, size, input_nc):
    w, h = pil_img.size
    s = size/h
    h = size
    w = int(s * w)
    img = np.array(pil_img.resize((w, h)))
    img = img.reshape(h, w, input_nc)
    return img, w, h

def load_train_data(image_path, load_size=286, fine_size_w=256, fine_size_h=256, input_nc=3, is_testing=False):
    img_A = Image.open(image_path[0]).convert('RGB')
    img_B = Image.open(image_path[1]).convert('RGB')
    if input_nc == 1:
        img_A = img_A.convert('L')
        img_B = img_B.convert('L')

    if not is_testing:
        img_A, w_a, h_a = scale_shorter(img_A, load_size)
        img_B, w_b, h_b = scale_shorter(img_B, load_size)
        h_a = int(np.ceil(np.random.uniform(0, h_a-fine_size_h)))
        w_a = int(np.ceil(np.random.uniform(0, w_a-fine_size_w)))
        h_b = int(np.ceil(np.random.uniform(0, h_b-fine_size_h)))
        w_b = int(np.ceil(np.random.uniform(0, w_b-fine_size_w)))
        img_A = img_A[h_a:h_a+fine_size_h, w_a:w_a+fine_size_w]
        img_B = img_B[h_b:h_b+fine_size_h, w_b:w_b+fine_size_w]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A, w_a, h_a = scale_shorter(img_A, fine_size_h) # Alert! Hard-coded
        img_B, w_b, h_b = scale_shorter(img_B, fine_size_h)
        h_a = int(np.ceil(np.random.uniform(0, h_a-fine_size_h)))
        w_a = int(np.ceil(np.random.uniform(0, w_a-fine_size_w)))
        h_b = int(np.ceil(np.random.uniform(0, h_b-fine_size_h)))
        w_b = int(np.ceil(np.random.uniform(0, w_b-fine_size_w)))
        img_A = img_A[h_a:h_a+fine_size_h, w_a:w_a+fine_size_w]
        img_B = img_B[h_b:h_b+fine_size_h, w_b:w_b+fine_size_w]

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=2)
    return img_AB

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
