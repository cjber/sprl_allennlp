import torch

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy

from typing import Dict, Optional


@Model.register('ner_lstm')
class NerLstm(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder

        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=num_labels)
        self.accuracy = CategoricalAccuracy()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embedded = self.embedder(tokens)
        encoded = self.encoder(embedded, mask)
        classified = self.classifier(encoded)

        output: Dict[str, torch.Tensor] = {}
        output['logits'] = classified

        if label is not None:
            output['loss'] = sequence_cross_entropy_with_logits(
                classified, label, mask)
            accuracy = self.accuracy(classified, label)
            output['accuracy'] = accuracy

        return output
