from xml.etree import ElementTree
from typing import List
from spacy.gold import biluo_tags_from_offsets

import spacy
import os
import tqdm


def extract_labels(text, tags):

    ent_labels = ['PATH', 'PLACE',
                  'SPATIAL_ENTITY']
    measure = ['MEASURE']

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
    return ents + measure


def spaceeval_to_conll(spaceeval_xml_file, nlp):
    root = ElementTree.parse(spaceeval_xml_file).getroot()

    text: str = root.find('TEXT').text
    tags: List = list(root.find('TAGS'))

    ents = extract_labels(text, tags)

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
    tokens = [t.text for t in doc]
    sents = [sent for sent in doc.sents]

    offset = 0
    sent_tokens = []
    sent_biluo = []
    for sent in sents:
        sent_tokens.extend(tokens[offset:offset+len(sent)])
        sent_tokens.extend('#')
        sent_biluo.extend(text_biluo[offset:offset+len(sent)])
        sent_biluo.extend('\n')
        offset += len(sent)

    f = list(zip(sent_tokens, sent_biluo))
    for pair in f:
        if '\n' in pair[0]:
            f.remove(pair)
    return f


nlp = spacy.load("en_core_web_lg")

with open('text.txt', 'w') as fp:
    for root_dir, _, file in list(os.walk('/home/cjber/data/sprl/ents_spaceeval/train/')):
        for data_file in tqdm.tqdm(file):
            if data_file.endswith('xml'):
                f = os.path.join(root_dir, data_file)
                conll_formatted = spaceeval_to_conll(f, nlp)
                fp.write('\n'.join(f'{x[0]}\t{x[1]}' for x in conll_formatted))
