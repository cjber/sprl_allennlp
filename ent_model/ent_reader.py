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
        nlp = spacy.load("en_core_web_lg")

        for root_dir, _, file in list(os.walk(file_path)):
            for data_file in tqdm.tqdm(file):
                if data_file.endswith('xml'):
                    f = os.path.join(root_dir, data_file)
                    root = ElementTree.parse(f).getroot()

                    text: str = root.find('TEXT').text
                    tags: List = list(root.find('TAGS'))

                    ents = self.extract_labels(text, tags)

                    # method for removing duplicate start and end values
                    # at present just keeps the first entity
                    temp = {}
                    for start, end, ent in ents:
                        if start not in temp and end not in temp:
                            temp[start] = (start, end, ent)
                            temp[end] = (start, end, ent)

                    ents = set(temp.values())
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

        ent_labels = ['PATH', 'PLACE',
                      'SPATIAL_ENTITY']
        measure = ['MEASURE']

        # later on will explore these
        # motion = ['MOTION']
        # sp_signals = ['SPATIAL_SIGNAL']
        # m_signals = ['MOTION_SIGNAL']
        # links = ['QSLINK', 'OLINK', 'MOVELINK', 'MEASURELINK']

        ents = [tag for tag in tags if tag.tag in ent_labels]
        ents = [(int(ent.attrib['start']),
                 int(ent.attrib['end']),
                 ent.tag + '_' + str(ent.attrib['form']))
                for ent in ents
                # very low accuracy for places without NAM/NOM
                if str(ent.attrib['form']) != ""]

        measure = [tag for tag in tags if tag.tag in measure]
        measure = [(int(ent.attrib['start']),
                    int(ent.attrib['end']),
                    ent.tag)
                   for ent in measure]

        motion = [tag for tag in tags if tag.tag in motion]
        motion = [(int(ent.attrib['start']),
                   int(ent.attrib['end']),
                   ent.tag)
                  for ent in motion]

        m_signals = [tag for tag in tags if tag.tag in m_signals]
        for m in m_signals:
            if 'motion_signal_type' in m.attrib:
                m.attrib['adjunct_type'] = m.attrib.pop('motion_signal_type')

        m_signals = [(int(ent.attrib['start']),
                      int(ent.attrib['end']),
                      ent.tag + "_" + str(ent.attrib["adjunct_type"]))
                     for ent in m_signals]

        sp_signals = [tag for tag in tags if tag.tag in sp_signals]
        sp_signals = [(int(ent.attrib['start']),
                       int(ent.attrib['end']),
                       ent.tag + "_" + str(ent.attrib["semantic_type"]))
                      for ent in sp_signals]

        return ents + measure + motion + m_signals + sp_signals

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


# t = SpaceEvalReader().read(
#     '/home/cjber/data/sprl/ents_spaceeval/train/')
