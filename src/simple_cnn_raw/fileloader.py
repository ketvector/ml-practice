from sys import byteorder
import numpy as np
from PIL import Image

print(byteorder)

def get_labels(filename, limit = None):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        #TODO assert magic is correct.
        num_items = int.from_bytes(f.read(4), "big")
        num_items = num_items if limit == None else limit
        labels = [0] * num_items
        for i in range(num_items):
            label = int.from_bytes(f.read(1), "big")
            labels[i] = label
        return labels

def get_images(filename, limit = None):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
            #TODO assert magic is correct.
        num_items = int.from_bytes(f.read(4), "big")
        num_items = num_items if limit == None else limit
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")
        print(magic, num_items, num_rows, num_cols)
        images = np.empty(shape=(num_items, num_rows, num_cols), dtype=np.float16)
        for i in range(num_items):
            img = np.empty(shape=(num_rows,num_cols), dtype=np.float16)
            for r in range(num_rows):
                for c in range(num_cols):
                    v = (float) (int.from_bytes(f.read(1), byteorder="big"))
                    img[r][c] = (v - 127.5)/ (127.5)
            images[i] = img
        return images
            

def get_data(limit = None):
    train_x = get_images("train-images.idx3-ubyte", limit)
    train_y = get_labels("train-labels.idx1-ubyte", limit)
    test_x = get_images("t10k-images.idx3-ubyte", limit)
    test_y = get_labels("t10k-labels.idx1-ubyte", limit)
    
    return (train_x, train_y, test_x, test_y)

        

        