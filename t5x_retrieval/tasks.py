# Copyright 2022 The T5X Retrieval Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add Tasks to registry."""
import functools

import seqio
import t5.data

tsv_english_path = {
        "train": "/home/mac/datasets/train_english.tsv",
        "validation": "/home/mac/datasets/dev_english.tsv",
}

tsv_german_path = {
        "train": "/home/mac/datasets/train_german.tsv",
        "validation": "/home/mac/datasets/dev_german.tsv",
}

tsv_arabic_path = {
        "train": "/home/mac/datasets/train_arabic.tsv",
        "validation": "/home/mac/datasets/dev_arabic.tsv",
}

tsv_chinese_path = {
        "train": "/home/mac/datasets/train_chinese.tsv",
        "validation": "/home/mac/datasets/dev_chinese.tsv",
}

tsv_dutch_path = {
        "train": "/home/mac/datasets/train_dutch.tsv",
        "validation": "/home/mac/datasets/dev_dutch.tsv",
}

tsv_french_path = {
        "train": "/home/mac/datasets/train_french.tsv",
        "validation": "/home/mac/datasets/dev_french.tsv",
}

tsv_hindi_path = {
        "train": "/home/mac/datasets/train_hindi.tsv",
        "validation": "/home/mac/datasets/dev_hindi.tsv",
}

tsv_indonesian_path = {
        "train": "/home/mac/datasets/train_indonesian.tsv",
        "validation": "/home/mac/datasets/dev_indonesian.tsv",
}

tsv_japanese_path = {
        "train": "/home/mac/datasets/train_japanese.tsv",
        "validation": "/home/mac/datasets/dev_japanese.tsv",
}

tsv_portuguese_path = {
        "train": "/home/mac/datasets/train_portuguese.tsv",
        "validation": "/home/mac/datasets/dev_portuguese.tsv",
}

tsv_russian_path = {
        "train": "/home/mac/datasets/train_russian.tsv",
        "validation": "/home/mac/datasets/dev_russian.tsv",
}

language_to_path = {
    "arabic": tsv_arabic_path,
    "english": tsv_english_path,
    "russian": tsv_russian_path,
    "portuguese": tsv_portuguese_path,
    "japanese": tsv_japanese_path,
    "hindi": tsv_hindi_path,
    "indonesian": tsv_indonesian_path,
    "dutch": tsv_dutch_path,
    "german": tsv_german_path,
    "chinese": tsv_chinese_path,
    "french": tsv_french_path,

}


# ================================== Negatives ==================================

tsv_english_negatives_path = {
        "train": "/home/mac/datasets/train_english_negatives.tsv",
        "validation": "/home/mac/datasets/dev_english.tsv",
}

tsv_german_negatives_path = {
        "train": "/home/mac/datasets/train_german_negatives.tsv",
        "validation": "/home/mac/datasets/dev_german.tsv",
}

tsv_arabic_negatives_path = {
        "train": "/home/mac/datasets/train_arabic_negatives.tsv",
        "validation": "/home/mac/datasets/dev_arabic.tsv",
}

tsv_chinese_negatives_path = {
        "train": "/home/mac/datasets/train_chinese_negatives.tsv",
        "validation": "/home/mac/datasets/dev_chinese.tsv",
}

tsv_dutch_negatives_path = {
        "train": "/home/mac/datasets/train_dutch_negatives.tsv",
        "validation": "/home/mac/datasets/dev_dutch.tsv",
}

tsv_french_negatives_path = {
        "train": "/home/mac/datasets/train_french_negatives.tsv",
        "validation": "/home/mac/datasets/dev_french.tsv",
}

tsv_hindi_negatives_path = {
        "train": "/home/mac/datasets/train_hindi_negatives.tsv",
        "validation": "/home/mac/datasets/dev_hindi.tsv",
}

tsv_indonesian_negatives_path = {
        "train": "/home/mac/datasets/train_indonesian_negatives.tsv",
        "validation": "/home/mac/datasets/dev_indonesian.tsv",
}

tsv_japanese_negatives_path = {
        "train": "/home/mac/datasets/train_japanese_negatives.tsv",
        "validation": "/home/mac/datasets/dev_japanese.tsv",
}

tsv_portuguese_negatives_path = {
        "train": "/home/mac/datasets/train_portuguese_negatives.tsv",
        "validation": "/home/mac/datasets/dev_portuguese.tsv",
}

tsv_russian_negatives_path = {
        "train": "/home/mac/datasets/train_russian_negatives.tsv",
        "validation": "/home/mac/datasets/dev_russian.tsv",
}

tsv_spanish_negatives_path = {
        "train": "/home/mac/datasets/train_spanish_negatives.tsv",
        "validation": "/home/mac/datasets/dev_spanish.tsv",
}

language_negatives_to_path = {
    "arabic": tsv_arabic_negatives_path,
    "english": tsv_english_negatives_path,
    "russian": tsv_russian_negatives_path,
    "portuguese": tsv_portuguese_negatives_path,
    "japanese": tsv_japanese_negatives_path,
    "hindi": tsv_hindi_negatives_path,
    "indonesian": tsv_indonesian_negatives_path,
    "dutch": tsv_dutch_negatives_path,
    "german": tsv_german_negatives_path,
    "chinese": tsv_chinese_negatives_path,
    "french": tsv_french_negatives_path,
    "spanish": tsv_spanish_negatives_path,
}


# ================================== CLirMatrix Pretraining ==================================

tsv_clirmatrix_multi_path = {
        "train": "/home/mac/datasets/train_clirmatrix.tsv",
        "validation": "/home/mac/datasets/dev_clirmatrix.tsv",
}



MULTILIGUAL_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"  # GCS
MULTILIGUAL_EXTRA_IDS = 100


def get_multilingual_vocabulary():
  return seqio.SentencePieceVocabulary(MULTILIGUAL_SPM_PATH)


DEFAULT_VOCAB = t5.data.get_default_vocabulary()
DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets":
        seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}


MULTILINGUAL_VOCAB = get_multilingual_vocabulary()
MULTILINGUAL_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(vocabulary=MULTILINGUAL_VOCAB, add_eos=True, required=False),
    "targets":
        seqio.Feature(vocabulary=MULTILINGUAL_VOCAB, add_eos=True)
}



MULTILINGUAL_OUTPUT_FEATURES_NEGATIVES = {
    "inputs":
        seqio.Feature(vocabulary=MULTILINGUAL_VOCAB, add_eos=True, required=False),
    "targets":
        seqio.Feature(vocabulary=MULTILINGUAL_VOCAB, add_eos=True),
    "negative_targets":
        seqio.Feature(vocabulary=MULTILINGUAL_VOCAB, add_eos=True),
}

# =========================== Fine-tuning Tasks/Mixtures =======================
# ----- Beir MS Marco-----
seqio.TaskRegistry.add(
    "beir_msmarco_retrieval",
    source=seqio.TfdsDataSource(
        tfds_name="beir/msmarco:1.0.0",
        splits={
            "train": "train",
            "validation": "validation",
        },
    ),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.rekey,
            key_map={
                "inputs": "query",
                "targets": "passage",
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES)
# ----- Beir MS Marco-----

# ----- Multilingual MS Marco-----

for language in list(language_to_path.keys()):
    seqio.TaskRegistry.add(
        f"mmarco_retrieval_{language}",
        source=seqio.TextLineDataSource(
            split_to_filepattern=language_to_path[language],
            ),
        preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["inputs","targets"]),
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
                ],
        metric_fns=[],
        output_features=MULTILINGUAL_OUTPUT_FEATURES,
    )



seqio.MixtureRegistry.add(
  "multilingual_marco_mixture",
  [(f"mmarco_retrieval_{language}", 1) for language in list(language_to_path.keys())]
)

# =============================== Multilingual MS Marco with Negatives ===============================
for language in list(language_to_path.keys()):
    seqio.TaskRegistry.add(
        f"mmarco_retrieval_{language}_negatives",
        source=seqio.TextLineDataSource(
            split_to_filepattern=language_negatives_to_path[language],
            ),
        preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["inputs","targets", "negative_targets"]),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[],
        output_features=MULTILINGUAL_OUTPUT_FEATURES_NEGATIVES,
    ) 

seqio.MixtureRegistry.add(
  "multilingual_marco_mixture_negatives",
  [(f"mmarco_retrieval_{language}_negatives", 1) for language in list(language_to_path.keys())]
)

# =============================== CLIRMatrix ===============================

seqio.TaskRegistry.add(
        f"clirmatrix_pretraining",
        source=seqio.TextLineDataSource(
            split_to_filepattern=tsv_clirmatrix_multi_path,
            ),
        preprocessors=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["inputs","targets"]),
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
                ],
        metric_fns=[],
        output_features=MULTILINGUAL_OUTPUT_FEATURES,
    )

# ============================ Inference Tasks/Mixtures =======================
# ----- Beir MS Marco-----
for split in ["query", "passage"]:
  seqio.TaskRegistry.add(
      f"beir_msmarco_retrieval_{split}",
      source=seqio.TfdsDataSource(
          tfds_name="beir/msmarco:1.0.0",
          splits={split: split},
      ),
      preprocessors=[
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  "inputs": split,
                  "targets": f"{split}_id",
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[],
      output_features=DEFAULT_OUTPUT_FEATURES)


# ----- Beir MS Marco-----
for split in ["query", "passage"]:
  seqio.TaskRegistry.add(
      f"mmarco_retrieval_de_{split}",
      source=seqio.TfdsDataSource(
          tfds_name="mrtydi/mmarco-en:1.0.0",
          splits={split: split},
      ),
      preprocessors=[
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  "inputs": split,
                  "targets": f"{split}_id",
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[],
      output_features=DEFAULT_OUTPUT_FEATURES)
