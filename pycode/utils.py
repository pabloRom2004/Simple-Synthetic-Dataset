# pycode/utils.py

from functools import wraps, lru_cache
import inspect
import tempfile
from pathlib import Path
from contextlib import contextmanager
from itertools import zip_longest
import re
import gzip
import numpy as np
import spacy
import shutil
import os
import sys
from urllib.request import urlretrieve
from config import FASTTEXT_EMBEDDINGS_DIR, SPECIAL_TOKEN_REGEX

def store_args(constructor):
    @wraps(constructor)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, 'args') or not hasattr(self, 'kwargs'):
            self.args = args
            self.kwargs = add_dicts(get_default_args(constructor), kwargs)
        return constructor(self, *args, **kwargs)
    return wrapped

def add_dicts(*dicts):
    return {k: v for dic in dicts for k, v in dic.items()}

def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

def get_temp_filepaths(n_filepaths, create=False):
    return [get_temp_filepath(create=create) for _ in range(n_filepaths)]

TEMP_DIR = None

def get_temp_filepath(create=False):
    global TEMP_DIR
    temp_filepath = Path(tempfile.mkstemp()[1])
    if TEMP_DIR is not None:
        temp_filepath.unlink()
        temp_filepath = TEMP_DIR / temp_filepath.name
        temp_filepath.touch(exist_ok=False)
    if not create:
        temp_filepath.unlink()
    return temp_filepath

def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float('inf')):
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert not strict, f'Files don\'t have the same number of lines: {filepaths}, use strict=False'
            if strip:
                parallel_lines = [l.rstrip('\n') if l is not None else None for l in parallel_lines]
            yield parallel_lines

@contextmanager
def write_lines_in_parallel(filepaths, strict=True):
    with open_files(filepaths, 'w') as files:
        yield FilesWrapper(files, strict=strict)

class FilesWrapper:
    def __init__(self, files, strict=True):
        self.files = files
        self.strict = strict

    def write(self, lines):
        assert len(lines) == len(self.files)
        for line, f in zip(lines, self.files):
            if line is None:
                assert not self.strict
                continue
            f.write(line.rstrip('\n') + '\n')

@contextmanager
def open_files(filepaths, mode='r'):
    files = []
    try:
        files = [Path(filepath).open(mode, encoding='utf-8') for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]

def remove_special_tokens(sentence):
    return extract_special_tokens(sentence)[1]

def extract_special_tokens(sentence):
    match = re.match(fr'(^(?:{SPECIAL_TOKEN_REGEX} *)+) *(.*)$', str(sentence))
    if match is None:
        return '', sentence
    special_tokens, sentence = match.groups()
    return special_tokens.strip(), sentence

def failsafe_division(a, b, default=0):
    if b == 0:
        return default
    return a / b

def yield_lines(filepath, gzipped=False, n_lines=None):
    filepath = Path(filepath)
    open_function = open
    if gzipped or filepath.name.endswith('.gz'):
        open_function = gzip.open
    with open_function(filepath, 'rt', encoding='utf-8') as f:
        for i, l in enumerate(f):
            if n_lines is not None and i >= n_lines:
                break
            yield l.rstrip('\n')

@lru_cache(maxsize=10)
def get_word2rank(vocab_size=10 ** 5, language='en'):
    word2rank = {}
    line_generator = yield_lines(get_fasttext_embeddings_path(language))
    next(line_generator)
    for i, line in enumerate(line_generator):
        if (i + 1) > vocab_size:
            break
        word = line.split(' ')[0]
        word2rank[word] = i
    return word2rank

def get_rank(word, language='en'):
    return get_word2rank(language=language).get(word, len(get_word2rank(language=language)))

def get_log_rank(word, language='en'):
    return np.log(1 + get_rank(word, language=language))

def get_log_ranks(text, language='en'):
    return [
        get_log_rank(word, language=language) for word in get_content_words(text, language=language) if word in get_word2rank(language=language)
    ]

def get_lexical_complexity_score(sentence, language='en'):
    log_ranks = get_log_ranks(sentence, language=language)
    if len(log_ranks) == 0:
        log_ranks = [np.log(1 + len(get_word2rank()))]
    return np.quantile(log_ranks, 0.75)

def get_dependency_tree_depth(sentence, language='en'):
    def get_subtree_depth(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max([get_subtree_depth(child) for child in node.children])

    tree_depths = [
        get_subtree_depth(spacy_sentence.root) for spacy_sentence in spacy_process(sentence, language=language).sents
    ]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)

def get_content_words(text, language='en'):
    return [token.text for token in get_spacy_content_tokens(text, language=language)]

def get_spacy_content_tokens(text, language='en'):
    def is_content_token(token):
        return not token.is_stop and not token.is_punct and token.ent_type_ == ''

    return [token for token in get_spacy_tokenizer()(text) if is_content_token(token)]

@lru_cache(maxsize=1)
def get_spacy_tokenizer():
    return spacy.load('en_core_web_md').tokenizer

@lru_cache(maxsize=10)
def get_spacy_model(language='en', size='md'):
    model_name = {
        'en': f'en_core_web_{size}',
        'fr': f'fr_core_news_{size}',
        'es': f'es_core_news_{size}',
        'it': f'it_core_news_{size}',
        'de': f'de_core_news_{size}',
    }[language]
    return spacy.load(model_name)

@lru_cache(maxsize=10 ** 6)
def spacy_process(text, language='en', size='md'):
    return get_spacy_model(language=language, size=size)(str(text))

def get_fasttext_embeddings_path(language='en'):
    fasttext_embeddings_path = Path(FASTTEXT_EMBEDDINGS_DIR) / f'cc.{language}.300.vec'
    if not fasttext_embeddings_path.exists():
        url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.vec.gz'
        fasttext_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(download_and_extract(url)[0], fasttext_embeddings_path)
    return fasttext_embeddings_path

def download_and_extract(url):
    tmp_dir = Path(tempfile.mkdtemp())
    compressed_filename = url.split('/')[-1]
    compressed_filepath = tmp_dir / compressed_filename
    download(url, compressed_filepath)
    print('Extracting...')
    extracted_paths = extract(compressed_filepath, tmp_dir)
    compressed_filepath.unlink()
    return extracted_paths

def download(url, destination_path=None, overwrite=True):
    if destination_path is None:
        destination_path = get_temp_filepath()
    if not overwrite and destination_path.exists():
        return destination_path
    print('Downloading...')
    try:
        urlretrieve(url, destination_path, reporthook)
        sys.stdout.write('\n')
    except (Exception, KeyboardInterrupt, SystemExit):
        print('Rolling back: remove partially downloaded file')
        os.remove(destination_path)
        raise
    return destination_path

def extract(filepath, output_dir):
    output_dir = Path(output_dir)
    # Infer extract function based on extension
    extensions_to_functions = {
        '.tar.gz': untar,
        '.tar.bz2': untar,
        '.tgz': untar,
        '.zip': unzip,
        '.gz': ungzip  # Add handling for .gz files
    }
    
    for extension, function in extensions_to_functions.items():
        if filepath.name.endswith(extension):
            return function(filepath, output_dir)
    raise ValueError(f'Unknown extension: {filepath.suffixes}')

def ungzip(filepath, output_dir):
    import gzip
    output_path = output_dir / filepath.stem  # removes .gz extension
    with gzip.open(filepath, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    return [output_path]

def untar(filepath, output_dir):
    import tarfile
    with tarfile.open(filepath, 'r:*') as tar:
        tar.extractall(path=output_dir)
    return list(output_dir.glob('*'))

def unzip(filepath, output_dir):
    import zipfile
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    return list(output_dir.glob('*'))

def reporthook(block_num, block_size, total_size):
    """
    Tracks the progress of the file download.
    """
    if total_size > 0:
        downloaded = block_num * block_size
        progress = min(100, (downloaded / total_size) * 100)
        sys.stdout.write(f"\rDownloading: {progress:.2f}%")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading: {block_num * block_size} bytes")
        sys.stdout.flush()