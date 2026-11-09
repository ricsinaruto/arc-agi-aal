"""Text-only observations for the Duck analyzer on a text-only model (letter-coded boards instead of images)."""

DUCK_IMAGE_RULE = (
    '- The newest user turn includes an image of the board as it was before this turn\'s '
    'actions (labelled with its step); earlier turns keep only a note that their image was omitted.\n'
)


def bind_duck(analyzer):
    if analyzer._system_prompt.count(DUCK_IMAGE_RULE) != 1:
        raise ValueError('Duck image prompt changed; cannot install text-only observations')
    analyzer._system_prompt = analyzer._system_prompt.replace(DUCK_IMAGE_RULE,
        '- The newest user turn includes an exact letter-coded settled board, with its step. '
        'Rows run top to bottom and columns left to right, both zero-based. '
        'The same symbols are used by current_frame.ascii and the evidence tools. '
        'No images are supplied; historical and intermediate frames remain available in Python.\n')
    original = analyzer._build_user_message

    def build(user_prompt, current_frame):
        if current_frame is None:
            raise ValueError('Text-only observation requires an authoritative frame')
        message = original(user_prompt, current_frame)
        parts = message.get('content')
        if not isinstance(parts, list) or sum(p.get('type') == 'image_url' for p in parts) != 1:
            raise ValueError('Unexpected Duck observation envelope')
        text = '\n'.join(p['text'] for p in parts if p['type'] == 'text')
        text = text.replace('Grid image of the board at step ', 'Grid text of the board at step ')
        return {**message, 'content': text + '\n' + current_frame.ascii}

    analyzer._build_user_message = build
