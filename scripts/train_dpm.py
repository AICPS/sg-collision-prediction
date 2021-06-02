import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
from baseline_risk_assessment.dpm_trainer import DPMTrainer

def train_dpm_model(args):
    trainer = DPMTrainer(args)
    trainer.learn()


if __name__ == "__main__":
    train_dpm_model(sys.argv[1:])
