import numpy as np
import cv2, os, struct

## TEST data
# Fashion-MNIST
test_labels_path = 't10k-labels-idx1-ubyte' 
test_images_path = 't10k-images-idx3-ubyte'
# MNIST
#test_labels_path = 'data/mnist/t10k-labels-idx1-ubyte'
#test_images_path = 'data/mnist/t10k-images-idx3-ubyte'

## TRAIN data
# Fashion-MNIST
train_labels_path = 'train-labels-idx1-ubyte'
train_images_path = 'train-images-idx3-ubyte'
# MNIST
#train_labels_path = 'data/mnist/train-labels-idx1-ubyte'
#train_images_path = 'data/mnist/train-images-idx3-ubyte'

def extract_mnist(images_path, labels_path, save_folder):
    '''
    Extract images of an idx byte file
    ----------
    images_path : filepath of the images (idx byte)
    labels_path : filepath of the labels (idx byte)
    save_folder : path for saving the raw images in subfolders of each class
    '''
    
    labels = []
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    with open(images_path, "rb") as images_file:
        images_file.read(4)
        images_file.read(4)
        images_file.read(4)
        images_file.read(4)
        count = 0
        image = np.zeros((28, 28, 1), np.uint8)
        image_bytes = images_file.read(784)
        while image_bytes:
            image_unsigned_char = struct.unpack("=784B", image_bytes)
            for i in range(784):
                image.itemset(i, image_unsigned_char[i])
            image_save_path = "%s/%i/%d.png" % (save_folder, labels[count], count)
            save_path = "%s/%i/" % (save_folder, labels[count])
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            cv2.imwrite(image_save_path, image)
            image_bytes = images_file.read(784)
            count += 1

# Test data saved in folder `data/test/fashion_mnist`
extract_mnist(test_images_path, test_labels_path,'test/fashion_mnist')

# Training data saved in folder `data/train/fashion_mnist`
extract_mnist(train_images_path, train_labels_path,'train/fashion_mnist')
