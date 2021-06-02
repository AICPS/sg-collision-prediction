import os, sys, pdb
import pickle as pkl
import cv2
from tqdm import tqdm
from pathlib import Path, PurePath
import multiprocessing
import torch
from collections import defaultdict
import numpy as np


#initializes globals for multiprocessing pool
def initializer(imsettings, outputdir, rescaleshape):
    global image_settings
    image_settings = imsettings
    global output_dir
    output_dir = outputdir
    global rescale_shape
    rescale_shape = rescaleshape

# TODO: Add honda support
#preprocesses all raw_images in a directory tree
def preprocess_directory(input_path=None, output_dir='dpm_images', image_settings=cv2.IMREAD_GRAYSCALE, rescale_shape=(64,64), num_processes=4):
    if(input_path is None):
        raise ValueError("please pass a valid input path.")
    all_video_clip_dirs = [x for x in input_path.iterdir() if x.is_dir()]
    all_video_clip_dirs = sorted(all_video_clip_dirs, key=lambda x: int(x.stem.split('_')[0]))
    pool = multiprocessing.Pool(num_processes, initializer, initargs=(image_settings, output_dir, rescale_shape))
    pool.map(preprocess_sequence, all_video_clip_dirs)
    print("Image preprocessing completed.")


#preprocesses a single sequence.
def preprocess_sequence(path):
    print("processing " + str(path))
    os.makedirs(str(path/output_dir), exist_ok=True)
    # read all frame numbers from raw_images. and store image_frames (list).
    raw_images = sorted(list(path.glob("raw_images/*.jpg")) +
                        list(path.glob("raw_images/*.png")), key=lambda x: int(x.stem))
    for raw_image_path in raw_images:
        frame = raw_image_path.stem
        image = cv2.imread(str(raw_image_path), image_settings)
        resized_image = cv2.resize(image, rescale_shape)
        cv2.imwrite(str(path/output_dir/frame)+".png", resized_image)
    return 1


#this class preprocesses labeled image data into input sequences for the DPM model.
class DPMPreprocessor():

    def __init__(self, input_path, cache_path="dpm_data.pkl", subseq_len=5, rescale_shape=(64,64), convert2gray=True, image_output_dir="dpm_images", num_processes=4):
        self.input_path = input_path
        self.cache_path = cache_path
        self.subseq_len = subseq_len
        self.rescale_shape = rescale_shape
        self.image_output_dir = image_output_dir
        self.num_processes = num_processes
        if convert2gray:
            self.image_settings = cv2.IMREAD_GRAYSCALE
        else:
            self.image_settings = cv2.IMREAD_UNCHANGED


    #preprocesses raw images into rescaled and recolored format for DPM.
    def preprocess_images(self):
        preprocess_directory(self.input_path, self.image_output_dir, self.image_settings, self.rescale_shape, self.num_processes)

    # TODO: Add honda support
    #load raw image data from directory
    def process_dataset(self):
        all_video_clip_dirs = [x for x in self.input_path.iterdir() if x.is_dir()]
        all_video_clip_dirs = sorted(all_video_clip_dirs, key=lambda x: int(x.stem.split('_')[0]))
        new_sequences = []
        for path in tqdm(all_video_clip_dirs):
            
            ignore_path = (path/"ignore.txt").resolve()
            if ignore_path.exists():
                with open(str(path/"ignore.txt"), 'r') as label_f:
                    ignore_label = int(label_f.read())
                    if ignore_label: continue;
            
            label_path = (path/"label.txt").resolve()

            if label_path.exists():
                with open(str(path/"label.txt"), 'r') as label_f:
                    risk_label = float(label_f.read().strip().split(",")[0])
            
                risk_label = 1 if risk_label >= 0 else 0 #binarize float value.
            else:
                print("Label not found for path: " + str(path))
                continue #skip paths that dont have labels

            subseqs, labels = self.process_sequence(path, risk_label)
            new_sequences.append((subseqs, labels, PurePath(path).name))

        with open(self.cache_path, 'wb') as f:
            pkl.dump(new_sequences, f, fix_imports=False)
    

    #generates a list of subsequences of length subseq_len from a top level sequence.
    def process_sequence(self, seq_path, label):
        # read all frame numbers from raw_images. and store image_frames (list).
        images = sorted(list(seq_path.glob(self.image_output_dir+"/*.jpg")) +
                            list(seq_path.glob(self.image_output_dir+"/*.png")), key=lambda x: int(x.stem))
        ims = []
        for image_path in images:
            ims.append(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)) #read images from file

        dim1 = len(ims) - self.subseq_len + 1 
        dim2 = self.subseq_len
        subseqs = np.zeros((dim1, dim2, self.rescale_shape[0], self.rescale_shape[1])) 
        labels = np.full((dim1), label)
        ims = np.array(ims) 

        #TODO optimize
        for i in range(dim1):
            subseqs[i,:] = ims[i:i+self.subseq_len] #fill array with subsequences of images

        return subseqs, labels
