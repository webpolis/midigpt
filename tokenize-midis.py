import sys
import os

from pathlib import Path
from miditok import REMI
from miditok.constants import ADDITIONAL_TOKENS
from miditoolkit import MidiFile
from tqdm import tqdm

do_BPE = False
path_suffix = '/bpe' if do_BPE else ''
midis_path = sys.argv[2] if len(sys.argv) > 2 else '/home/nico/data/midis'
TOKENS_PATH = sys.argv[1] if len(
    sys.argv) > 1 else '/home/nico/data/ai/models/midi'
token_params_path = Path(f"{TOKENS_PATH}{path_suffix}/token_params.json")
midi_paths = list(Path(midis_path).glob('*.mid'))

pitch_range = range(21, 109)
additional_tokens = ADDITIONAL_TOKENS
additional_tokens['Chord'] = True
additional_tokens['TimeSignature'] = True
additional_tokens['Rest'] = True
# additional_tokens['time_signature_range'] = (8, 2)

tokenizer = REMI(pitch_range=pitch_range,
                 additional_tokens=additional_tokens)

print('Tokenizing dataset...')
for midi_path in tqdm(midi_paths):
    try:
        midi = MidiFile(midi_path)
        tokens = tokenizer(midi_path)
        midi_name = os.path.basename(midi_path)

        tokenizer.save_tokens(tokens, f'{TOKENS_PATH}/{midi_name}.json')
    except Exception as e:
        pass

if do_BPE:
    # Constructs the vocabulary with BPE, from the tokenized files
    print('Learning BPE...')
    tokenizer.learn_bpe(
        vocab_size=len(tokenizer),
        tokens_paths=Path(TOKENS_PATH).glob('*.json'),
        start_from_empty_voc=False,
    )

    # Converts the tokenized musics into tokens with BPE
    print('Applying BPE...')
    tokenizer.apply_bpe_to_dataset(
        Path(f'{TOKENS_PATH}/'), Path(f'{TOKENS_PATH}{TOKENS_PATH}{path_suffix}'))

print('Saving params...')
tokenizer.save_params(token_params_path)
