import json
import xml.etree.ElementTree as ET
from convert_locomo import convert_conversation, prettify

with open('c:/Users/Liam/Desktop/Struct_Memory_R1/locomo_structured_data/locomo10.json') as f:
    data = json.load(f)

conv26_data = next(c for c in data if c['sample_id'] == 'conv-26')
root_el = convert_conversation(conv26_data)

speakers = root_el.findall('.//Speaker')
print(f"Number of Speakers: {len(speakers)}")
for s in speakers:
    obs_count = len(s.findall('.//Observation'))
    print(f"Speaker: {s.get('name')}, Observations: {obs_count}")
