// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{"type":"spaceeval_reader"},
  "train_data_path": "/home/cjber/drive/phd/programming/sprl_allennlp/sprl/train.txt",
  "validation_data_path": "/home/cjber/drive/phd/programming/sprl_allennlp/sprl/test.txt",
  "model": {
    "type": "sprl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
            "trainable": true
        }
      }
    },
    "initializer": {
      "regexes": [
        ["tag_projection_layer.*weight", { "type": "orthogonal" }]
      ]
    },
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 200,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_highway": true
    },
    "binary_feature_dim": 100
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 80
    }
  },

  "trainer": {
    "num_epochs": 500,
    "grad_clipping": 1.0,
    "patience": 20,
    "validation_metric": "+f1-measure-overall",
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
