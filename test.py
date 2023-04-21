from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from miditok.constants import CHORD_MAPS, ADDITIONAL_TOKENS
from pathlib import Path

# Creates the tokenizer and loads a MIDI
pitch_range = range(21, 109)
additional_tokens = ADDITIONAL_TOKENS
# additional_tokens['Chord'] = True
additional_tokens['Rest'] = True
additional_tokens['TimeSignature'] = True

tokenizer = REMI(pitch_range=pitch_range,
                     additional_tokens=additional_tokens)
midi_path = '/home/nico/data/midis/1.mid'
midi = MidiFile(midi_path)
#programs = get_midi_programs(midi)

# Converts MIDI to tokens, and back to a MIDI
# calling it will automatically detect MIDIs, paths and tokens before the conversion
tokens = tokenizer(midi, apply_bpe_if_possible=False)

# Converts MIDI files to tokens saved as JSON files
""" midi_paths = list(Path("path", "to", "dataset").glob("**/*.mid"))
data_augmentation_offsets = [2, 1, 1]  # data augmentation on 2 pitch octaves, 1 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_paths, Path("path", "to", "tokens_noBPE"),
                                data_augment_offsets=data_augmentation_offsets) """


# Saving our tokenizer, to retrieve it back later with the load_params method
token_params_path = Path("/home/nico/data/ai/models/midi/token_params.json")
tokens_path = Path("/home/nico/data/ai/models/midi/tokens.json")

# Converts the tokenized musics into tokens with BPE
# tokenizer.apply_bpe_to_dataset(Path('/home/nico/data/ai/models/midi/'), Path('path', 'to', 'tokens_BPE'))

# Constructs the vocabulary with BPE, from the tokenized files
""" tokenizer.learn_bpe(
    vocab_size=500,
    tokens_paths=list([tokens_path]),
    start_from_empty_voc=False,
)
tokenizer.apply_bpe(tokens) """

tokenizer.save_tokens(tokens=tokens, path=tokens_path)
tokenizer.save_params(token_params_path)
