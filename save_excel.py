import pandas
import os

def create_excel_files(output_path, filename):
    data = []
    dataset = "Balance_800"
    embed_dim = 300
    relation_dim = 300
    gat_dim = 256
    num_dim = True
    txt_dim = True
    columns = ["Aggregator", "Number Layers", "Learning Rate", "Dropout", "Convolutional Dim", "Batch Size", "Best Pretrain", "Best Finetune", "Accuracy"]

    lr_list = [0.0001, 0.001, 0.01, 0.1]
    batch_size_list = [2048]
    dropout_list = [ 0.1, 0.5]
    n_layer_list = [2, 4, 8]
    n_dim_list = [16, 32]
    

    for i in batch_size_list:
        for j in n_dim_list:
            for k in dropout_list:
                for v in lr_list:
                    for x in n_layer_list:
                        default_best_pretrain = -1
                        default_best_finetune = -1
                        default_accuracy = 0
                        aggregator = "gcn"

                        source_path = f"trained_model/LiteralKG/{dataset}/embed-dim{embed_dim}_relation-dim{relation_dim}_{aggregator}_n-layers{x}_gat{gat_dim}_conv{j}_bs{i}_num{num_dim}_txt{txt_dim}_lr{v}_dropout{k}_pretrain0/run/"

                        # Loop all the folder to get the log files
                        for path, subdirs, files in os.walk(source_path):
                            for name in files:
                                file_name = os.path.join(path, name)
                                filename_split = file_name.split(".pth")[0].split("pre-training_model_epoch")
                                print(filename_split[0])
                                
                                if len(filename_split) > 1:
                                    try:
                                        if default_best_pretrain < int(filename_split[1]):
                                            default_best_pretrain = int(filename_split[1])
                                    except:
                                        pass

                                else:
                                    filename_split = filename.split(".")[0].split("training_model_epoch")
                                    
                                    if len(filename_split) > 1:
                                        try:
                                            if default_best_finetune < int(filename_split[1]):
                                                default_best_finetune = int(filename_split[1])
                                        except:
                                            pass

                        case = f"{aggregator} {x} {v} {k} {j} {i} {default_best_pretrain} {default_best_finetune} {default_accuracy}"
                        row = case.split(" ")
                        data.append(row)

    df = pandas.DataFrame(data, columns=columns)

    df.to_excel(f'{output_path}/{filename}', sheet_name='data')


def main():
    output_path = "outputs"
    filename = "test.xlsx"

    create_excel_files(output_path, filename)

if __name__ == '__main__':
    main()
