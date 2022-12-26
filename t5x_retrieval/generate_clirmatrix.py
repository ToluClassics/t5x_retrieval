import pandas as pd
import ir_datasets
from tqdm import tqdm

data = []
for src_lang in tqdm(["ar", "de", "en", "es", "fr", "ja", "ru", "zh"]):
    for tgt_lang in ["ar", "de", "en", "es", "fr", "ja", "ru", "zh"]:

        if src_lang != tgt_lang:
            dataset = ir_datasets.load(f"clirmatrix/{src_lang}/multi8/{tgt_lang}/train")
            docstore = dataset.docs_store()

            queries = {}
            for query in dataset.queries_iter():
                queries[query.query_id] = query.text

            for qrels in dataset.qrels_iter():
                if qrels.relevance in [3,4,5,6]:
                    data.append([queries[qrels.query_id], docstore.get(qrels.doc_id).text])

print("writing to file")
output_train_df = pd.DataFrame(data)
output_train_df.to_csv(f"datasets/train_clirmatrix.tsv", sep = "\t", header=False, index=False)