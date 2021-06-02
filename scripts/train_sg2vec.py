import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from sg_risk_assessment.sg2vec_trainer import SG2VECTrainer
import pandas as pd


def train_sg2vec_model(args, iterations=1):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''
    
    outputs = []
    labels = []
    metrics = []

    for i in range(iterations):
        trainer = SG2VECTrainer(args)
        trainer.split_dataset()
        trainer.build_model()
        trainer.learn()
        if trainer.config.n_folds <= 1:
            outputs_train, labels_train, outputs_test, labels_test, metric = trainer.evaluate()

            outputs += outputs_test
            labels  += labels_test
            metrics.append(metric)

    if len(outputs) and len(labels) and len(metrics):
        # Store the prediction results. 
        store_path = trainer.config.cache_path.parent
        outputs_pd = pd.DataFrame(outputs)
        labels_pd  = pd.DataFrame(labels)
        
        labels_pd.to_csv(store_path / "dynkg_training_labels.tsv", sep='\t', header=False, index=False)
        outputs_pd.to_csv(store_path / "dynkg_training_outputs.tsv", sep="\t", header=False, index=False)
        
        # Store the metric results. 
        metrics_pd = pd.DataFrame(metrics[-1]['test'], index=[0])
        metrics_pd.to_csv(store_path / "dynkg_classification_metrics.csv", header=True)


if __name__ == "__main__":
    # the entry of dynkg pipeline training
    train_sg2vec_model(sys.argv[1:])
