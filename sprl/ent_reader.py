import os
from typing import Dict, Iterable, List
from xml.etree import ElementTree

import spacy
import tqdm
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, SequenceLabelField, TextField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.predictors.predictor import Predictor
from spacy.gold import biluo_tags_from_offsets
from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans


@DatasetReader.register('ents_spaceeval')
class SpaceEvalReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = SpacyTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 dep_predictor: Predictor = None,
                 ** kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or Tokenizer()
        self.token_indexers = token_indexers or {'tokens': TokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:

        instances = []
        nlp = spacy.load("en_core_web_sm")

        for root_dir, _, file in list(os.walk(file_path)):
            for data_file in tqdm.tqdm(file):
                if data_file.endswith('xml'):
                    f = os.path.join(root_dir, data_file)
                    root = ElementTree.parse(f).getroot()

                    text: str = root.find('TEXT').text
                    tags: List = list(root.find('TAGS'))

                    # set ensures no duplicates
                    ents = set(self.extract_labels(text, tags))
                    doc = nlp(text)

                    text_biluo = biluo_tags_from_offsets(doc, ents)
                    # for some reason some tags are considered incorrect
                    # can't see the problem but it needs to be handled
                    try:
                        bioul_tags_to_spans(text_biluo)
                    except:
                        continue
                    tokens = [t.text for t in doc]
                    instance = self.text_to_instance(
                        tokens,
                        text_biluo)
                    instances.append(instance)
                elif data_file.endswith('txt'):
                    f = os.path.join(root_dir, data_file)
                    with open(f) as txt_file:
                        for line in txt_file:
                            tokens = self.tokenizer.tokenize(line)
                            instance = self.text_to_instance(
                                tokens)
                            instances.append(instance)
        yield from instances

    def extract_labels(self, text, tags):
        ent_labels = ['PATH', 'PLACE', 'MOTION',
                      'NONMOTION_EVENT', 'SPATIAL_ENTITY',
                      'MEASURE']

        ents = [tag for tag in tags if tag.tag in ent_labels]
        return [(int(ent.attrib['start']),
                       int(ent.attrib['end']), ent.tag)
                      for ent in ents]

    def text_to_instance(self,
                         tokens: List[str],
                         ents: List[str] = None) -> Instance:

        if ents:
            tokens = [Token(w) for w in tokens]
            text_field = TextField(tokens, self.token_indexers)
            label_field = SequenceLabelField(ents, text_field)
            fields: Dict[str, Field] = {'tokens': text_field,
                                        'tags': label_field}
        else:
            text_field = TextField(tokens, self.token_indexers)
            fields: Dict[str, Field] = {'tokens': text_field}
        return Instance(fields)

file_path = '/home/cjber/data/sprl/ents_spaceeval/train/'

