local data_dir = "/home/cjber/data/sprl/ents_spaceeval";
local lstm_input_dim = 1202;
local ff_dim = 400;
local lstm_hidden_dim = 200;
local batch_size = 8;
local cuda_device = 0;
local num_epochs = 10;
local seed = 42;

local char_embedding_dim = std.parseInt(std.extVar('char_embedding_dim'));
local ff_num_layers = std.parseInt(std.extVar('ff_num_layers'));
local lstm_num_layers = std.parseInt(std.extVar('lstm_num_layers'));
local dropout = std.parseJson(std.extVar('dropout'));
local lr = std.parseJson(std.extVar('lr'));

{
    numpy_seed: seed,
    pytorch_seed: seed,
    random_seed: seed,
     dataset_reader: {
         type: "sprl.ent_reader.SpaceEvalReader",
         token_indexers: {
             elmo: {
                 type: "elmo_characters"
            },
             token_characters: {
                 type: "characters"
            },
             tokens: {
                 type: "single_id",
                 lowercase_tokens: true
            }
        }
    },
     train_data_path: data_dir + "/train/",
     validation_data_path: data_dir + "/test/",
     data_loader: {
         batch_sampler: {
             type: "bucket",
             batch_size: batch_size # 64 optimal
        }
    },
     model: {
         type: "allennlp_models.tagging.models.crf_tagger.CrfTagger",
         calculate_span_f1: true,
         label_encoding: "BIOUL",
         dropout: dropout,
         verbose_metrics: true,
         encoder: {
             type: "stacked_bidirectional_lstm",
             hidden_size: lstm_hidden_dim,
             input_size: lstm_input_dim,
             num_layers: lstm_num_layers,
             recurrent_dropout_probability: 0.5,
             use_highway: true
        },
         feedforward: {
             activations: "tanh",
             dropout: dropout, 
             hidden_dims: ff_dim,
             input_dim: ff_dim,
             num_layers: ff_num_layers
        },
         include_start_end_transitions: false,
         initializer: {
           regexes: [
            [
                ".*tag_projection_layer.*weight",
                {
                     type: "xavier_uniform"
                }
            ],
            [
                ".*tag_projection_layer.*bias",
                {
                     type: "zero"
                }
            ],
            [
                ".*feedforward.*weight",
                {
                     type: "xavier_uniform"
                }
            ],
            [
                ".*feedforward.*bias",
                {
                     type: "zero"
                }
            ],
            [
                ".*weight_ih.*",
                {
                     type: "xavier_uniform"
                }
            ],
            [
                ".*weight_hh.*",
                {
                     type: "orthogonal"
                }
            ],
            [
                ".*bias_ih.*",
                {
                     type: "zero"
                }
            ],
            [
                ".*bias_hh.*",
                {
                     type: "lstm_hidden_bias"
                }
            ]
          ]
        },
         regularizer: {
           regexes: [
            [
                "scalar_parameters",
                {
                     alpha: 0.001,
                     type: "l2"
                }
            ]
          ]
        },
         text_field_embedder: {
           token_embedders: {
             elmo: {
                 type: "elmo_token_embedder",
                 do_layer_norm: false,
                 dropout: 0
            },
             token_characters: {
                 type: "character_encoding",
                 embedding: {
                     embedding_dim: char_embedding_dim,
                     sparse: true,
                     vocab_namespace: "token_characters"
                },
                 encoder: {
                     type: "lstm",
                     hidden_size: 128,
                     input_size: char_embedding_dim,
                     num_layers: 1
                }
            },
             tokens: {
                 type: "embedding",
                 embedding_dim: 50,
                 pretrained_file: "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
                 sparse: true,
                 trainable: true
            }
          }
        }
    },
     trainer: {
         cuda_device: cuda_device,
         grad_norm: 5,
         num_epochs: num_epochs,
         optimizer: {
             type: "dense_sparse_adam",
             lr: 0.001
        },
         patience: 25,
         validation_metric:  "+f1-measure-overall"
    }
}
