import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run LiteralKG.")

    parser.add_argument('--exp_name', type=str, default="run")
    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='Balance_800',
                        help='Choose a dataset')
    parser.add_argument('--data_dir', nargs='?', default='data/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='data/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='fine-tuning_model_epoch15.pth',
                        help='Path of stored model.')

    parser.add_argument('--fine_tuning_batch_size', type=int, default=2048,
                        help='Fine Tuning batch size.')
    parser.add_argument('--pre_training_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help='Test batch size (the head number to test every batch).')

    parser.add_argument('--total_ent', type=int, default=1000,
                        help='Total entities.')
    parser.add_argument('--total_rel', type=int, default=100,
                        help='Total relations.')

    parser.add_argument('--embed_dim', type=int, default=300,
                        help='head / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=300,
                        help='Relation Embedding size.')
    parser.add_argument('--scale_gat_dim', type=int, default=256,
                        help='Scale gat concatenation.')
    parser.add_argument('--num_lit_dim', type=int, default=2,
                        help='Numerical Literal Embedding size.')
    parser.add_argument('--txt_lit_dim', type=int, default=300,
                        help='Text Literal Embedding size.')

    parser.add_argument('--use_num_lit', type=bool, default=True,
                        help='Using Numerical Literal Embedding.')
    parser.add_argument('--use_txt_lit', type=bool, default=True,
                        help='Using Text Literal Embedding.')

    parser.add_argument('--laplacian_type', type=str, default='random-walk',

                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction, gin}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[32, 32, 32, 32, 32, 32, 32, 32, 32]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--n_conv_layers', type=int, default=9,
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--fine_tuning_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating Fine Tuning l2 loss.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--milestone_score', type=float, default=0.5,
                        help='The condition of link score.')

    parser.add_argument('--n_epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--epoch_data_rate', type=float, default=1,
                        help='Sampling data rate for each epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--fine_tuning_print_every', type=int, default=500,
                        help='Iter interval of printing Fine Tuning loss.')
    parser.add_argument('--kg_print_every', type=int, default=500,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating Fine Tuning.')

    # parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
    #                     help='Calculate metric@K when evaluating.')

    parser.add_argument('--pre_training_neg_rate', type=int, default=3,
                        help='The pre-training negative rate.')
    parser.add_argument('--fine_tuning_neg_rate', type=int, default=3,
                        help='The fine tuning negative rate.')
    parser.add_argument('--test_neg_rate', type=int, default=1,
                        help='The fine tuning negative rate.')
    parser.add_argument('--train_data_rate', type=int, default=0.8,
                        help='The (train data/all train data) rate. Validate data rate = 1 - train_data_rate.')


    parser.add_argument('--device', nargs='?', default='cuda:0',
                        help='Choose a device to run')
    parser.add_argument('--prediction_dict_file', nargs='?', default='disease_dict.pickle',
                        help='Disease dictionary file')

    parser.add_argument('--use_residual', type=bool, default=True,
                        help='Use residual connection.')

    parser.add_argument('--use_parallel_gpu', type=bool, default=False,
                        help='Use many GPUs.')

    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')

    parser.add_argument('--n_mlp_layers', type=int, default=3, help='The number of GIN hidden layer.')
    parser.add_argument('--mlp_hidden_dim', type=int, default=64, help='The dimension of GIN hidden layer.')

    args = parser.parse_args()

    args.data_name = args.data_name.replace("'", "")

    save_dir = 'trained_model/LiteralKG/{}/embed-dim{}_relation-dim{}_{}_n-layers{}_gat{}_num{}_txt{}_lr{}_pretrain0/{}/'.format(
        args.data_name, args.embed_dim, args.relation_dim, args.aggregation_type,
        args.n_conv_layers, args.scale_gat_dim, args.use_num_lit, args.use_txt_lit, args.lr,
        args.exp_name)
    args.save_dir = save_dir
    args.pretrain_model_path = f"{args.save_dir}{args.pretrain_model_path}"

    return args

