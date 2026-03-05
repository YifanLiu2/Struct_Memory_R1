import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import os

def sanitize_text(text):
    """Escape special XML characters in text."""
    if text is None:
        return ""
    return str(text)

def get_session_number(key):
    """Extract integer session number from keys like 'session_3' or 'session_3_observation'."""
    match = re.search(r'session_(\d+)', key)
    return int(match.group(1)) if match else None

def normalize_obs_key(key):
    """Handle the malformed '   ' key in conv-26 which should be session_1_observation."""
    if key.strip() == '':
        return 1
    return get_session_number(key)

def convert_conversation(conv_data):
    """Convert a single LoCoMo conversation dict to an XML Element tree."""
    
    conv = conv_data['conversation']
    observations = conv_data.get('observation', {})
    event_summary = conv_data.get('event_summary', {})
    session_summary = conv_data.get('session_summary', {})
    sample_id = conv_data.get('sample_id', 'unknown')

    speaker_a = conv.get('speaker_a', 'Speaker_A')
    speaker_b = conv.get('speaker_b', 'Speaker_B')

    # Root element
    conversation_el = ET.Element('Conversation', attrib={
        'sample_id': sample_id,
        'speaker_a': speaker_a,
        'speaker_b': speaker_b
    })

    # ----------------------------------------------------------------
    # Build Speaker nodes with Observations
    # ----------------------------------------------------------------
    # Collect all observations per speaker across all sessions
    speaker_observations = {speaker_a: [], speaker_b: []}

    for obs_key, obs_val in observations.items():
        session_num = normalize_obs_key(obs_key)
        if session_num is None:
            continue
        for speaker, claims in obs_val.items():
            if speaker not in speaker_observations:
                speaker_observations[speaker] = []
            for claim in claims:
                # claim is [text, source_dia_id]
                if isinstance(claim, list) and len(claim) == 2:
                    speaker_observations[speaker].append({
                        'text': claim[0],
                        'source_dia_id': claim[1],
                        'session_id': str(session_num)
                    })

    for speaker_name in [speaker_a, speaker_b]:
        speaker_el = ET.SubElement(conversation_el, 'Speaker', attrib={
            'name': speaker_name
        })
        for obs in speaker_observations.get(speaker_name, []):
            ET.SubElement(speaker_el, 'Observation', attrib={
                'text': sanitize_text(obs['text']),
                'source_dia_id': sanitize_text(obs['source_dia_id']),
                'session_id': sanitize_text(obs['session_id'])
            })

    # ----------------------------------------------------------------
    # Build Session nodes
    # ----------------------------------------------------------------
    # Find all session numbers that have turn data
    session_keys_with_turns = [
        k for k in conv.keys()
        if re.match(r'^session_\d+$', k) and isinstance(conv[k], list)
    ]
    session_numbers = sorted(
        [get_session_number(k) for k in session_keys_with_turns]
    )

    for sn in session_numbers:
        datetime_val = conv.get(f'session_{sn}_date_time', '')
        session_el = ET.SubElement(conversation_el, 'Session', attrib={
            'session_id': str(sn),
            'datetime': sanitize_text(datetime_val)
        })

        # Summary node
        summary_text = session_summary.get(f'session_{sn}_summary', '')
        if summary_text:
            summary_el = ET.SubElement(session_el, 'Summary')
            summary_el.text = sanitize_text(summary_text)

        # Event nodes
        event_key = f'events_session_{sn}'
        if event_key in event_summary:
            event_data = event_summary[event_key]
            for speaker_name in [speaker_a, speaker_b]:
                events = event_data.get(speaker_name, [])
                for event_text in events:
                    ET.SubElement(session_el, 'Event', attrib={
                        'speaker': speaker_name,
                        'description': sanitize_text(event_text)
                    })

        # Turn nodes
        turns = conv.get(f'session_{sn}', [])
        for turn in turns:
            turn_attribs = {
                'dia_id': sanitize_text(turn.get('dia_id', '')),
                'speaker': sanitize_text(turn.get('speaker', '')),
                'text': sanitize_text(turn.get('text', ''))
            }
            caption = turn.get('blip_caption', '')
            if caption:
                turn_attribs['image_caption'] = sanitize_text(caption)
            ET.SubElement(session_el, 'Turn', attrib=turn_attribs)

    return conversation_el


def prettify(element):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent='  ')


def main():
    input_path = '/mnt/user-data/uploads/locomo10.json'
    output_dir = '/mnt/user-data/outputs'
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path) as f:
        data = json.load(f)

    print(f"Converting {len(data)} conversations...")

    # Write one XML file per conversation
    for conv_data in data:
        sample_id = conv_data.get('sample_id', 'unknown')
        conversation_el = convert_conversation(conv_data)
        xml_str = prettify(conversation_el)
        out_path = os.path.join(output_dir, f'{sample_id}.xml')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        print(f"  Written: {out_path}")

    # Also write a single combined file
    root = ET.Element('LoCoMo')
    for conv_data in data:
        conversation_el = convert_conversation(conv_data)
        root.append(conversation_el)
    
    combined_xml = prettify(root)
    combined_path = os.path.join(output_dir, 'locomo10_structured.xml')
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write(combined_xml)
    print(f"\nCombined file written: {combined_path}")

    # Print stats
    print("\n--- Conversion Stats ---")
    for conv_data in data:
        conv = conv_data['conversation']
        sample_id = conv_data['sample_id']
        speaker_a = conv['speaker_a']
        speaker_b = conv['speaker_b']
        session_keys = [k for k in conv.keys() if re.match(r'^session_\d+$', k) and isinstance(conv[k], list)]
        n_sessions = len(session_keys)
        n_turns = sum(len(conv[k]) for k in session_keys)
        obs = conv_data.get('observation', {})
        n_obs = sum(
            len(claims)
            for obs_val in obs.values()
            for claims in obs_val.values()
        )
        print(f"  {sample_id} ({speaker_a} & {speaker_b}): {n_sessions} sessions, {n_turns} turns, {n_obs} observations")


if __name__ == '__main__':
    main()
