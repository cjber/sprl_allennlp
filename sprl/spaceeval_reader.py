import os
import pandas as pd
from typing import Dict, Iterable, List
from xml.etree import ElementTree

import spacy
from spacy.gold import biluo_tags_from_offsets
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, SequenceLabelField, TextField
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.predictors.predictor import Predictor
from allennlp.data.token_indexers import TokenIndexer
import tqdm


@DatasetReader.register('sprl_spaceeval')
class SpaceEvalReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 dep_predictor: Predictor = None,
                 ** kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or Tokenizer()
        self.token_indexers = token_indexers or {'tokens': TokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:

        instances = []

        for root_dir, _, file in list(os.walk(file_path)):
            for data_file in tqdm.tqdm(file):
                if not data_file.endswith('xml'):
                    continue
                f = os.path.join(root_dir, data_file)
                tree = ElementTree.parse(f)
                root = tree.getroot()
                text = root.find('TEXT').text
                tags = list(root.find('TAGS'))

                relations = self.extract_labels(text, tags)
                nlp = spacy.blank("en")
                nlp.add_pipe(nlp.create_pipe("sentencizer"))
                doc = nlp(text)

                for relation in relations:
                    rel_biluo = biluo_tags_from_offsets(doc, relation)
                    offset = 0
                    sentences = [sent for sent in doc.sents]
                    for sent in sentences:
                        biluo = rel_biluo[offset:offset+len(sent)]
                        offset += len(sent)
                        tokens = [t.text for t in sent]
                        instance = self.text_to_instance(
                            tokens,
                            biluo
                        )
                        if biluo.count('O') != len(biluo):
                            instances.append(instance)

        for instance in instances:
            yield instance

    def extract_labels(self, text, tags):
        # LANDMARKS
        ent_labels = ['PATH', 'PLACE', 'MOTION',
                      'NONMOTION_EVENT', 'SPATIAL_ENTITY',
                      'MEASURE']
        ents = [tag for tag in tags if tag.tag in ent_labels]

        ent_id = pd.DataFrame(
            {'id_ent': [ent.attrib['id'] for ent in ents],
             'start': [int(ent.attrib['start']) for ent in ents],
             'end': [int(ent.attrib['end']) for ent in ents]})

        # SPATIAL INDICATORS (as V in SRL)
        spatial_inds = ['SPATIAL_SIGNAL', 'MOTION_SIGNAL']
        spatial_indicators = [tag for tag in tags if tag.tag in spatial_inds]

        ind_df = pd.DataFrame(
            {'id_ent': [ind.attrib['id'] for ind in spatial_indicators],
             'start': [int(ind.attrib['start']) for ind in spatial_indicators],
             'end': [int(ind.attrib['end']) for ind in spatial_indicators]}
        )
        ent_id = ent_id.append(ind_df)

        # SPATIAL RELATIONS
        sprl = ['QSLINK', 'OLINK']
        spatial_relations = [tag for tag in tags if tag.tag in sprl]

        sr_dict = pd.DataFrame(
            {'id_sr': [sr.attrib['id'] for sr in spatial_relations],
             'rel_type': [sr.attrib['relType'] for sr in spatial_relations],
             'TR': [sr.attrib['trajector'] for sr in spatial_relations],
             'LM': [sr.attrib['landmark'] for sr in spatial_relations],
             'SI': [sr.attrib['trigger'] for sr in spatial_relations]}
        )

        def merge_ents_to_sr(ent_id, sr_dict, actor):
            relations = ent_id.merge(sr_dict[['id_sr', actor]],
                                     left_on='id_ent',
                                     right_on=actor, how='outer'
                                     )\
                .dropna()
            return relations

        sent_labels = []
        for _, row in sr_dict.iterrows():
            # some SRs have TR and LM for single entity
            if row['TR'] == row['LM']:
                row['LM'] = ''
            elif row['TR'] == row['SI']:
                row['SI'] = ''
            relations = []
            for rel in ['TR', 'LM', 'SI']:
                start = ent_id[ent_id['id_ent'] == row[rel]]['start']
                end = ent_id[ent_id['id_ent'] == row[rel]]['end']
                if start.empty:
                    continue
                relations.append(
                    (int(start.values),
                     int(end.values), rel)
                )
            sent_labels.append(relations)
        return sent_labels

    def text_to_instance(self,
                         tokens: List[str],
                         ents: List[str] = None,
                         ent_id: List[str] = None,
                         sr_dict: List[Dict[str, str]] = None) -> Instance:
        tokens = [Token(w) for w in tokens]
        text_field = TextField(tokens, self.token_indexers)
        label_field = SequenceLabelField(ents, text_field)

        fields: Dict[str, Field] = {'tokens': text_field,
                                    'label': label_field}
        return Instance(fields)
