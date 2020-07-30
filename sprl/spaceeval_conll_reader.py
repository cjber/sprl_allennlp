import itertools
from typing import Dict, Iterator, List

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides
from spacy.gold import biluo_tags_from_offsets


def _is_divider(line: str) -> bool:
    return line.strip() == ''


@DatasetReader.register('spaceeval_reader')
class SpaceevalConllReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()
        }

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as conll_file:
            for divider, lines in itertools.groupby(conll_file, _is_divider):
                # skip over any dividing lines
                if divider:
                    continue
                # get the CoNLL fields, each token is a list of fields
                fields = [line.strip().split() for line in lines]
                # switch it so that each field is a list of tokens/labels
                fields = [line for line in zip(*fields)]

                # only keep the tokens and NER labels
                tokens, ner_tags = fields
                yield self.text_to_instance(tokens, ner_tags)

    @overrides
    def text_to_instance(self,
                         words: List[str],
                         ner_tags: List[str]) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField([Token(w) for w in words], self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["tags"] = SequenceLabelField(ner_tags, tokens)

        return Instance(fields)
