import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import os
from collections import defaultdict


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
    """Convert a single LoCoMo conversation dict to an XML Element tree.

    Structure: Conversation -> Speaker -> Session -> {Summary, Observation}
    """
    conv = conv_data['conversation']
    observations = conv_data.get('observation', {})
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
    # Collect observations per speaker per session
    # ----------------------------------------------------------------
    speaker_session_obs = defaultdict(lambda: defaultdict(list))

    for obs_key, obs_val in observations.items():
        session_num = normalize_obs_key(obs_key)
        if session_num is None:
            continue
        for speaker, claims in obs_val.items():
            for claim in claims:
                if isinstance(claim, list) and len(claim) == 2:
                    speaker_session_obs[speaker][session_num].append({
                        'text': claim[0],
                        'source_dia_id': claim[1],
                    })

    # ----------------------------------------------------------------
    # Collect session datetimes and summaries
    # ----------------------------------------------------------------
    session_keys_with_turns = [
        k for k in conv.keys()
        if re.match(r'^session_\d+$', k) and isinstance(conv[k], list)
    ]
    all_session_numbers = sorted(
        [get_session_number(k) for k in session_keys_with_turns]
    )

    session_datetimes = {}
    session_summaries = {}
    for sn in all_session_numbers:
        session_datetimes[sn] = conv.get(f'session_{sn}_date_time', '')
        session_summaries[sn] = session_summary.get(f'session_{sn}_summary', '')

    # ----------------------------------------------------------------
    # Build Speaker -> Session -> {Summary, Observation} hierarchy
    # ----------------------------------------------------------------
    for speaker_name in [speaker_a, speaker_b]:
        speaker_el = ET.SubElement(conversation_el, 'Speaker', attrib={
            'name': speaker_name
        })

        # Get all sessions where this speaker has observations
        speaker_sessions = sorted(speaker_session_obs.get(speaker_name, {}).keys())

        for sn in speaker_sessions:
            datetime_val = session_datetimes.get(sn, '')
            summary_text = session_summaries.get(sn, '')
            session_attribs = {
                'session_id': str(sn),
                'datetime': sanitize_text(datetime_val)
            }
            if summary_text:
                session_attribs['summary'] = sanitize_text(summary_text)
                
            session_el = ET.SubElement(speaker_el, 'Session', attrib=session_attribs)

            # Observation children (use child elements, not attributes)
            for obs in speaker_session_obs[speaker_name][sn]:
                obs_el = ET.SubElement(session_el, 'Observation')
                obs_text_el = ET.SubElement(obs_el, 'text')
                obs_text_el.text = sanitize_text(obs['text'])
                obs_src_el = ET.SubElement(obs_el, 'source_dia_id')
                obs_src_el.text = sanitize_text(obs['source_dia_id'])

    # Wrap in version structure: Root -> Conversation_Version -> content
    root_el = ET.Element('Root')
    version_el = ET.SubElement(root_el, 'Conversation_Version', attrib={'number': '1'})
    patch_el = ET.SubElement(version_el, 'patch_info')
    patch_el.text = 'None'
    conv_hist_el = ET.SubElement(version_el, 'conversation_history')
    conv_hist_el.text = 'Initial_Version'

    # Move all conversation children into the version node
    for child in list(conversation_el):
        version_el.append(child)
    # Copy conversation attributes to version node for reference
    for key, val in conversation_el.attrib.items():
        version_el.set(key, val)

    return root_el


def prettify(element):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent='  ')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'locomo_structured_data', 'locomo10.json')
    output_dir = os.path.join(script_dir, 'locomo_structured_data')
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"Converting {len(data)} conversations...")
    print(f"Structure: Conversation -> Speaker -> Session -> Observation")
    print()

    # Write one XML file per conversation
    for conv_data in data:
        sample_id = conv_data.get('sample_id', 'unknown')
        conversation_el = convert_conversation(conv_data)
        xml_str = prettify(conversation_el)
        out_path = os.path.join(output_dir, f'{sample_id}.xml')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        print(f"  Written: {out_path}")

    # Print stats
    print("\n--- Conversion Stats ---")
    for conv_data in data:
        conv = conv_data['conversation']
        sample_id = conv_data['sample_id']
        speaker_a = conv['speaker_a']
        speaker_b = conv['speaker_b']
        session_keys = [k for k in conv.keys() if re.match(r'^session_\d+$', k) and isinstance(conv[k], list)]
        n_sessions = len(session_keys)
        obs = conv_data.get('observation', {})
        n_obs = sum(
            len(claims)
            for obs_val in obs.values()
            for claims in obs_val.values()
        )
        # Count obs per speaker
        speaker_obs_count = defaultdict(int)
        for obs_val in obs.values():
            for speaker, claims in obs_val.items():
                speaker_obs_count[speaker] += len(claims)
        obs_breakdown = ", ".join(f"{s}: {c}" for s, c in speaker_obs_count.items())
        print(f"  {sample_id} ({speaker_a} & {speaker_b}): {n_sessions} sessions, {n_obs} observations ({obs_breakdown})")


if __name__ == '__main__':
    main()
