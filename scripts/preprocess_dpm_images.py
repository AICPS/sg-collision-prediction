import os, sys, pdb
sys.path.append(os.path.dirname(sys.path[0]))
from baseline_risk_assessment.dpm_preprocessor import DPMPreprocessor
from argparse import ArgumentParser
from pathlib import Path

#This script runs pre-processing of image data for use in the DPM pipeline
def preprocess_dpm_data(args):
    parser = ArgumentParser()
    parser.add_argument("--input_path",type=str,default='M:/louisccc/av/synthesis_data/legacy_dataset/lane-change-100-balanced',help='directory containing the raw data sequences and labels.')
    parser.add_argument("--cache_path", type=str, default='dpm_data.pkl', help="path to save processed sequence data.")
    parser.add_argument("--subseq_len", type=int, default=5, help="length of output subsequences")
    parser.add_argument("--preprocess", help="use this option to preprocess images before subsequencing.")
    parser.add_argument("--rescale_shape", type=str, default="64,64", help="reshaped images will be this size. Format: x,y ")
    parser.add_argument("--image_output_dir", type=str, default="dpm_images_64x64", help="directory where processed images will be saved.")
    parser.add_argument("--grayscale", help="use this option to convert images to grayscale during processing.")
    parser.add_argument("--num_processes", type=int, default=4, help="number of processes to run in parallel")
    config = parser.parse_args(args)
    config.input_path = Path(config.input_path).resolve()
    config.cache_path = Path(config.cache_path).resolve()
    config.rescale_shape = (int(config.rescale_shape.split(',')[0]), int(config.rescale_shape.split(',')[1]))
    preprocessor = DPMPreprocessor(config.input_path, 
                                    config.cache_path, 
                                    config.subseq_len, 
                                    config.rescale_shape, 
                                    config.grayscale, 
                                    config.image_output_dir, 
                                    config.num_processes)
    if(config.preprocess):
        preprocessor.preprocess_images()
    preprocessor.process_dataset()

if __name__ == "__main__":
    preprocess_dpm_data(sys.argv[1:])