from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance


@Predictor.register('sprl_predictor')
class SprlPredictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')

        outputs['tokens'] = [str(token)
                             for token in instance.fields['tokens'].tokens]
        outputs['predicted'] = [label_vocab[logit]
                                for logit in outputs['logits'].argmax(1)]
        return sanitize(outputs)
