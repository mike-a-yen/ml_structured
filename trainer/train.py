import argparse

from datasets.raw_dataset import RawDataset
from datasets.processed_dataset import ProcessedDataset
from datasets.featurized_dataset import FeaturizedDataset
from datasets.trainable_dataset import TrainableDataset

from models.base_model import BaseModel
from models.rnn_model import RNNModel
from trainer.model_handler import ModelHandler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subsample',default=0,type=int)
    parser.add_argument('--max-epochs',default=8,type=int)
    args,_ = parser.parse_known_args()
    return args

def main() -> None:

    args = parse_args()
    raw_dataset = RawDataset(subsample=args.subsample)
    print(f'Raw dataset has {len(raw_dataset.raw_df)} records.')

    processed_dataset = ProcessedDataset(raw_dataset)
    print(f'Processed dataset has {len(processed_dataset.processed_df)} records.')

    featurized_dataset = FeaturizedDataset(processed_dataset)
    print(f'Featurized dataset has {len(featurized_dataset.featurized_df)} records.')

    model = RNNModel(featurized_dataset)
    handler = ModelHandler(model,use_wandb=True)
    print(f'Comensing training on {handler.device}')
    handler.fit(args.max_epochs)
    print('Done.')



if __name__ == '__main__':
    main()
