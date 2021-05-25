import os, sys
import check_gpu as cg
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
sys.path.append(os.path.dirname(sys.path[0]))
from DPM.dpm_trainer import DPMTrainer

def train_dpm_model(args):
    trainer = DPMTrainer(args)
    trainer.learn()


if __name__ == "__main__":
    train_dpm_model(sys.argv[1:])
