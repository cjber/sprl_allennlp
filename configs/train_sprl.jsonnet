// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        "type": "sprl_spaceeval",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "train_data_path": "/home/cjber/data/sprl/spaceeval/Traning/",
    "validation_data_path": "/home/cjber/data/sprl/spaceeval/spaceeval_trial_data/",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "padding_noise": 0.1,
            "batch_size": 8
        }
    },
  "model": {
    "type": 'ner_lstm',
    "embedder": {
      "token_embedders": {
        "tokens": {
        "type": 'embedding',
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.lower.converted.zip",
          "embedding_dim": 300,
          "trainable": false
        }
      }
    },
    "encoder": {
      "type": 'alternating_lstm',
      "num_layers": 10,
      "input_size": 300,
      "hidden_size": 25
    }
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "grad_clipping": 5.0,
    "validation_metric": '-loss',
    "optimizer": {
      "type": 'adam',
      "lr": 0.003
    }
  }
}
