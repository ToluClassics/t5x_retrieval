import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from datasets import load_dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lang', type=str)
args = parser.parse_args()

language = args.lang


collection_dataset = load_dataset("unicamp-dl/mmarco", f"collection-{language}")
queries_dataset = load_dataset("unicamp-dl/mmarco", f"queries-{language}")


collection_dict = collection_dataset["collection"].to_dict()
queries_train_dict = queries_dataset["train"].to_dict()
queries_dev_dict = queries_dataset["dev"].to_dict()

collection_dict = dict(zip(collection_dict['id'], collection_dict['text']))
queries_train_dict = dict(zip(queries_train_dict['id'], queries_train_dict['text']))
queries_dev_dict = dict(zip(queries_dev_dict['id'], queries_dev_dict['text']))

dev_df = pd.read_csv("dumps/qrels.dev.small.tsv", header=None, sep="\t")
dev_dict = {}

for index, rows in dev_df.iterrows():
    dev_dict[queries_dev_dict[rows[0]]] = collection_dict[rows[2]]

output_dev_df = pd.DataFrame(list(dev_dict.items()))
output_dev_df.to_csv(f"datasets/dev_{language}.tsv", sep = "\t", header=False, index=False)


def get_data(idx, row):
    return [queries_train_dict[row[0]], collection_dict[row[1]]]


def main():
    train_df = pd.read_csv("dumps/train.tsv", header=0, sep="\t")
    train_dict = {}
    print('Extract query, collection pair')
    results = [get_data(idx, row) for idx, row in tqdm(train_df.iterrows(), total = 532752)]
    print('Sort items into list')
    print('write to file')
    output_train_df = pd.DataFrame(results)
    output_train_df.to_csv(f"datasets/train_{language}.tsv", sep = "\t", header=False, index=False)


if __name__ == "__main__":
    main()
