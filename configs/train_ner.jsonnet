// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        // This name needs to match the name that you used to register your dataset reader, with
        // the call to `@DatasetReader.register()`.
        "type": "sprl_spaceeval",
        // These other parameters exactly match the constructor parameters of your dataset reader class.
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "/home/cjber/data/sprl/spaceeval/spaceeval_trial_data/Copala.xml",
    "validation_data_path": "/home/cjber/data/sprl/spaceeval/spaceeval_trial_data/47_N_22_E.xml",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "padding_noise": 0.1,
            "batch_size": 48
        }
    },
  model: {
    type: 'ner_lstm',
    embedder: {
      token_embedders: {
        tokens: {
        type: 'embedding',
          pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.300d.txt",
          embedding_dim: 300,
          trainable: false
        }
      }
    },
    encoder: {
      type: 'gru',
      num_layers: 100,
      input_size: 300,
      hidden_size: 25
    }
  },
  trainer: {
    num_epochs: 40,
    patience: 10,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
}
