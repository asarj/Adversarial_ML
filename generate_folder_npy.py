import os
import fnmatch
import numpy as np

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
    return result

def generate(path='paht_to_vimeo90k/vimeo_septuplet/sequences/'):
    folder = find('im1.png', path)
    np.save('folder.npy', folder)