from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from miditok.constants import CHORD_MAPS, ADDITIONAL_TOKENS
from pathlib import Path

token_params_path = Path("/home/nico/data/ai/models/midi/token_params.json")
tokens_path = Path("/home/nico/data/ai/models/midi/")
midi_paths = list(Path('/home/nico/data/midis').glob('*.mid'))

pitch_range = range(21, 109)
additional_tokens = ADDITIONAL_TOKENS
# additional_tokens['Chord'] = True
additional_tokens['TimeSignature'] = True
# additional_tokens['Rest'] = True
# additional_tokens['Chord'] = True
# additional_tokens['time_signature_range'] = (8, 2)

tokenizer = REMI(pitch_range=pitch_range,
                 additional_tokens=additional_tokens)

tokenizer.tokenize_midi_dataset(        # 2 velocity and 1 duration values
    midi_paths,
    tokens_path,
)

tokenizer.save_params(token_params_path)

# midi = MidiFile(midi_path)
#programs = get_midi_programs(midi)

# Converts the tokenized musics into tokens with BPE
# tokenizer.apply_bpe_to_dataset(Path('/home/nico/data/ai/models/midi/'), Path('path', 'to', 'tokens_BPE'))

# Constructs the vocabulary with BPE, from the tokenized files
""" tokenizer.learn_bpe(
    vocab_size=len(tokenizer),
    tokens_paths=list([tokens_path]),
    start_from_empty_voc=False,
)
tokenizer.apply_bpe(tokens) """
