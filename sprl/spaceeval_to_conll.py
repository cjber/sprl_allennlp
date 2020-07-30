import os
import pandas as pd
from xml.etree import ElementTree
from typing import List

import spacy
import tqdm
from allennlp.data.dataset_readers.dataset_utils.span_utils import \
    bioul_tags_to_spans
from spacy.gold import biluo_tags_from_offsets


def extract_labels(tags, sent, offset):
    ent_labels = ['PATH', 'PLACE',
                  'SPATIAL_ENTITY']
    ents = [tag for tag in tags if tag.tag in ent_labels
            if int(tag.attrib['end']) <= len(sent.text) + offset
            if int(tag.attrib['start']) >= offset]
    ents = [(int(ent.attrib['start']) - offset,
             int(ent.attrib['end']) - offset,
             ent.tag + '_' + str(ent.attrib['form']))
            for ent in ents
            # very low accuracy for places without NAM/NOM
            if str(ent.attrib['form']) != ""]

    measure = ['MEASURE']
    measure = [tag for tag in tags if tag.tag in measure
               if int(tag.attrib['end']) <= len(sent.text) + offset
               if int(tag.attrib['start']) >= offset]

    measure = [(int(ent.attrib['start']) - offset,
                int(ent.attrib['end']) - offset,
                ent.tag)
               for ent in measure]

    return ents + measure


def coreference_res():
    # data has METALINK with relType = COREFERENCE
    pass


def create_triplets(tags, sent, offset):
    # from dsouza2015:
    # (1) each triplet contains a trajector, landmark, and trigger
    # (2) neither the trajector or landmark are type spatial-signal or
    # motion signal
    # (3) The trigger is a spatial-signal
    # ONE landmark/trajector per triple may be IMPLICIT
    # label a training instance as TRUE if these elements form a correct triple
    # LANDMARKS

    ent_labels = ['PATH', 'PLACE', 'SPATIAL_ENTITY']
    ents = [tag for tag in tags if tag.tag in ent_labels]

    ent_id = pd.DataFrame(
        {'id_ent': [ent.attrib['id'] for ent in ents],
         'start': [int(ent.attrib['start']) - offset for ent in ents],
         'end': [int(ent.attrib['end']) - offset for ent in ents]})

    ent_id = ent_id[(ent_id['start'] >= offset) &
                    (ent_id['end'] <= offset + len(sent.text))]

    # SPATIAL INDICATORS (as V in SRL)
    spatial_inds = ['SPATIAL_SIGNAL', 'MOTION_SIGNAL']
    spatial_indicators = [tag for tag in tags if tag.tag in spatial_inds]
    ind_df = pd.DataFrame(
        {'id_ent': [ind.attrib['id'] for ind in spatial_indicators],
         'start': [int(ind.attrib['start']) - offset
                   for ind in spatial_indicators],
         'end': [int(ind.attrib['end']) - offset
                 for ind in spatial_indicators]}
    )
    ind_df = ind_df[(ind_df['start'] >= offset) &
                    (ind_df['end'] <= offset + len(sent.text))]
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


def spaceeval_to_conll(spaceeval_xml_file, nlp):
    root = ElementTree.parse(spaceeval_xml_file).getroot()

    text: str = root.find('TEXT').text
    tags: List = list(root.find('TAGS'))

    doc = nlp(text)

    offset = 0
    sent_tokens = []
    sent_ents = []
    sent_sprl = []
    for sent in doc.sents:
        import ipdb;ipdb.set_trace()
        sent_nlp = nlp(sent.text)
        tokens = [str(token) for token in sent_nlp]
        spatial_entities = extract_labels(tags, sent, offset)
        spatial_triplets = create_triplets(tags, sent, offset)
        spatial_triplets = [triplet for triplet in spatial_triplets if triplet]

        ent_biluo = biluo_tags_from_offsets(sent_nlp, spatial_entities)
        sprl_biluo = []
        for triplet in spatial_triplets:
            sprl_biluo.append(biluo_tags_from_offsets(sent_nlp, triplet))

        if sprl_biluo == []:
            sent_sprl.extend(['O'] * len(tokens))
            sent_sprl.append('\n')
            sent_tokens.extend(tokens)
            sent_tokens.append('')
            sent_ents.extend(ent_biluo)
            sent_ents.append('\n')
        else:
            # duplicate for multiple sprls
            for sprl in sprl_biluo:
                sent_tokens.extend(tokens)
                sent_tokens.append('')
                sent_ents.extend(ent_biluo)
                sent_ents.append('\n')
                sent_sprl.extend(sprl)
                sent_sprl.append('')
        offset += len(sent.text)

    file_conll = list(zip(sent_tokens, sent_ents, sent_sprl))

    for pair in file_conll:
        if '\n' in pair[0]:
            file_conll.remove(pair)
        elif '\u2002' in pair[0]:
            file_conll.remove(pair)
        elif ' ' in pair[0]:
            file_conll.remove(pair)
    return file_conll


nlp = spacy.load("en_core_web_lg")

with open('train.txt', 'w') as fp:
    for root_dir, _, file in list(
            os.walk('/home/cjber/data/sprl/ents_spaceeval/train/')
    ):
        for data_file in tqdm.tqdm(file):
            if data_file.endswith('xml'):
                f = os.path.join(root_dir, data_file)
                conll_formatted = spaceeval_to_conll(f, nlp)
                fp.write('\n'.join(
                    f'{x[0]} {x[1]} {x[2]}'
                    for x in conll_formatted))

with open('test.txt', 'w') as fp:
    for root_dir, _, file in list(
            os.walk('/home/cjber/data/sprl/ents_spaceeval/test/')
    ):
        for data_file in tqdm.tqdm(file):
            if data_file.endswith('xml'):
                f = os.path.join(root_dir, data_file)
                conll_formatted = spaceeval_to_conll(f, nlp)
                fp.write('\n'.join(
                    f'{x[0]} {x[1]}'
                    for x in conll_formatted))
