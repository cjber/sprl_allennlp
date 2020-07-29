# import os
# from xml.etree import ElementTree

# f = '/home/cjber/data/sprl/ents_spaceeval/train/CP/45_N_22_E.xml'
# root = ElementTree.parse(f).getroot()

# text: str = root.find('TEXT').text
# tags = list(root.find('TAGS'))

# ents = ['PLACE']
# spans = [(int(tag.attrib['start']), int(tag.attrib['end']), tag.tag)
#          for tag in tags
#          if tag.tag in ents]

# tokens = [(text[start:end], tag) for start, end, tag in spans]
