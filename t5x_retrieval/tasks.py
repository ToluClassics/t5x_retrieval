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
seqio.TaskRegistry.add(
    "mmarco_retrieval_de",
    source=seqio.TfdsDataSource(
        tfds_name="mrtydi/mmarco-de:1.0.0",
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
    output_features=MULTILINGUAL_OUTPUT_FEATURES)

seqio.TaskRegistry.add(
    "mmarco_retrieval_en",
    source=seqio.TfdsDataSource(
        tfds_name="mrtydi/mmarco-en:1.0.0",
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
    output_features=MULTILINGUAL_OUTPUT_FEATURES)


seqio.MixtureRegistry.add(
  "multilingual_marco_mixture",
  [("mmarco_retrieval_de", 1), ("mmarco_retrieval_en", 1)]
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
