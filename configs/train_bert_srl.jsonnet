{
    "dataset_reader" : {
        "type": "sprl_spaceeval",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "/home/cjber/data/sprl/spaceeval/Traning/",
    "validation_data_path": "/home/cjber/data/sprl/spaceeval/spaceeval_trial_data/",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "padding_noise": 0.1,
            "batch_size": 48
        }
    },
  "model": {
    "type": 'srl_bert',
    "bert_model": 'bert-base-uncased'
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
