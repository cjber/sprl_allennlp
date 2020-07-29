# import itertools
# from allennlp.data.tokenizers import Token
# from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
# from allennlp.data.instance import Instance
# from allennlp.data.fields import Field, TextField, SequenceLabelField
# from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# from overrides import overrides
# from typing import Dict, List, Iterator


# def is_divider(line):
#     return line.strip() == ''


# @DatasetReader.register("lrec20_reader")
# class Lrec20Reader(DatasetReader):
#     def __init__(self,
#                  token_indexers: Dict[str, TokenIndexer] = None,
#                  lazy: bool = False) -> None:
#         super().__init__(lazy)
#         self._token_indexers = token_indexers or {
#             'tokens': SingleIdTokenIndexer()}

#     @overrides
#     def _read(self, file_path: str) -> Iterator[Instance]:
#         with open(file_path, 'r') as lrec_file:
#             # itertools.groupby is a powerful function that can group
#             # successive items in a list by the returned function call.
#             # In this case, we're calling it with `is_divider`, which returns
#             # True if it's a blank line and False otherwise.
#             for divider, lines in itertools.groupby(lrec_file, is_divider):
#                 # skip over any dividing lines
#                 if divider:
#                     continue
#                 # get the CoNLL fields, each token is a list of fields
#                 fields = [l.strip().split() for l in lines]
#                 # switch it so that each field is a list of tokens/labels
#                 fields = [l for l in zip(*fields)]
#                 # only keep the tokens and NER labels
#                 tokens, ner_tags = fields

#                 new_ner_tags = self.convert_tags_to_isospace(ner_tags)

#                 yield self.text_to_instance(tokens, new_ner_tags)

#     @overrides
#     def text_to_instance(self,
#                          words: List[str],
#                          ner_tags: List[str]) -> Instance:
#         fields: Dict[str, Field] = {}
#         # wrap each token in the file with a token object
#         tokens = TextField([Token(w) for w in words], self._token_indexers)

#         # Instances in AllenNLP are created using Python dictionaries,
#         # which map the token key to the Field type
#         fields["tokens"] = tokens
#         fields["label"] = SequenceLabelField(ner_tags, tokens)

#         return Instance(fields)

#     def convert_tags_to_isospace(self,
#                                  tags: List[str]):
#         # look at paper to convert to the ISOSpace specification
#         for tag in tags:
#             if "-DIF" in tag:
#                 tag == 'O'
#             # etc
#         return tags


# file_path = "/home/cjber/data/sprl/lrec20/voa/annotations/imf.conll.txt.anno"
# dr = Lrec20Reader()
# ins = dr.read(file_path)

# for i in ins:
#     print(i)
