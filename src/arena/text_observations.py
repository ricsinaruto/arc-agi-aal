"""Exact, explicitly labelled text evidence for the approved text-only comparison."""

from pathlib import Path

from qwen_baselines.evidence import grid_text

DUCK_IMAGE_RULE = (
    '- The newest user turn includes an image of the board as it was before this turn\'s '
    'actions (labelled with its step); earlier turns keep only a note that their image was omitted.\n'
)
PI_IMAGE_RULE = (
    'The initial settled image is supplied once. Later images and grids are on-demand\n'
    'immutable references. Use read on an image to view it.'
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


def pi_api(text):
    if text.count(PI_IMAGE_RULE) != 1:
        raise ValueError('Pi image prompt changed; cannot install text-only observations')
    return text.replace(PI_IMAGE_RULE,
        'The initial settled board is supplied once as exact text. Later grids are on-demand\n'
        'immutable references: read grid_text or load grid_npy in Python. Symbols 0-9,A-F\n'
        'encode color IDs 0-15, with rows top-to-bottom and columns left-to-right, zero-based.\n'
        'No images can be viewed. Reading a recorded frame PNG returns its exact text grid instead.')


def pi_system(text):
    from qwen_baselines.prompts import legend_line

    image_legend = legend_line(text_grid=False, image_pixels=512)
    if text.count(image_legend) != 1:
        raise ValueError('Pi system legend changed; cannot install text-only observations')
    return text.replace(image_legend, legend_line(text_grid=True, image_pixels=512)) + (
        '\nNo images can be viewed. The initial exact settled text board is supplied once; '
        'later evidence remains available through immutable grid_text and grid_npy references.\n')


def pi_initial(archive):
    return '\n\nExact initial settled board (0-9,A-F = color IDs 0-15):\n' + grid_text(archive.frame())


def recorded_grid_path(archive, workspace, path):
    """Only substitute immutable recorded PNGs, never arbitrary user-created images."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(workspace) / path
    resolved = path.resolve()
    evidence = (archive.root / 'evidence').resolve()
    if not resolved.is_relative_to(evidence):
        raise PermissionError('Text-only image access requires this game\'s recorded evidence; use Python for other files')
    # Match the exact export naming contract; retain the usual executor isolation check.
    if (resolved.suffix != '.png' or not resolved.stem.startswith('frame-')
            or not resolved.parent.name.startswith('obs-') or not resolved.is_file()
            or not resolved.with_suffix('.txt').is_file()):
        raise ValueError('No recorded exact grid for this image')
    return str(resolved.with_suffix('.txt'))
