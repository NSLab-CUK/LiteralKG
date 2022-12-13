import torch
import numpy as np
from dataloader import DataLoader
from time import time
from model import LiteralKG
import pandas as pd

from argument import parse_args


from utils.log_utils import *
from utils.metric_utils import *
from utils.model_utils import *

def test_model(args):
    device = torch.device(args.device)

    # load data
    data = DataLoader(args, logging)
    torch.cuda.empty_cache()

    # construct model & optimizer
    model = LiteralKG(args, data.n_entities,
                 data.n_relations, data.A_in, data.num_embedding_table, data.text_embedding_table)

    model = load_model(model, args.pretrain_model_path)
    model.to(device)
    time1 = time()

    prediction_scores, metrics_dict = evaluate(model, data.test_head_dict, data.test_batch_size, data.prediction_tail_ids, device, neg_rate=args.test_neg_rate)

    metrics_str = 'Running test: Total Time {:.1f}s | Accuracy [{:.4f}], Precision [{:.4f}], Recall [{:.4f}], F1 [{:.4f}]'.format(
        time() - time1, metrics_dict['accuracy'], metrics_dict['precision'], metrics_dict['recall'], metrics_dict['f1'])
    temp_metrics_df = pd.DataFrame(data=[{"metrics": metrics_str}])
    temp_metrics_df.to_csv(
        args.save_dir + '/test_results.tsv', sep='\t', index=False)
        
    np.save(args.save_dir + 'prediction_scores.npy', prediction_scores)
    print(metrics_str)

def main():
    args = parse_args()
    test_model(args)


if __name__ == '__main__':
    main()
