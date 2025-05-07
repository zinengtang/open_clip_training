from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, InterpolationMode,  ColorJitter, RandomApply, RandomGrayscale, Normalize
from torch.utils.data import Dataset
from itertools import cycle
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
import requests
from dataclasses import field
from typing import List
import ast
import json
import logging
import math
import os
import random
import io
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from torchvision import transforms
def decode_image_bytes(byte_data):
    """
    Decode the bytes back into a PIL Image.
    """
    img_byte_arr = io.BytesIO(byte_data)
    pil_image = Image.open(img_byte_arr)
    return pil_image


import os
import requests
from io import BytesIO
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
from PIL import Image

import nltk
from nltk.tokenize import sent_tokenize
import spacy
from itertools import combinations
import random
class TextSegmenter:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
    def get_full_sentences(self, text):
        """Extract complete sentences from the text."""
        return sent_tokenize(text)
    
    def get_noun_phrases(self, text):
        """Extract noun phrases using spaCy."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    
    def get_sentence_combinations(self, text, num_sentences=2):
        """Generate combinations of complete sentences."""
        sentences = sent_tokenize(text)
        return ['. '.join(combo) + '.' for combo in combinations(sentences, num_sentences)]
    
    def get_cross_sentence_segments(self, text):
        """Create segments by combining parts from different sentences."""
        doc = self.nlp(text)
        segments = []
        
        # Split into clauses based on punctuation and conjunctions
        clauses = []
        current_clause = []
        
        for token in doc:
            current_clause.append(token.text)
            if token.text in ['.', ',', 'and', 'but', 'or']:
                if current_clause:
                    clauses.append(' '.join(current_clause))
                current_clause = []
        
        if current_clause:
            clauses.append(' '.join(current_clause))
            
        # Generate random combinations of clauses
        for i in range(len(clauses)):
            for j in range(i + 2, len(clauses)):
                segment = ' '.join(clauses[i:j])
                if len(segment.split()) >= 5:  # Minimum 5 words for meaningfulness
                    segments.append(segment)
                    
        return segments
    
    def get_subject_predicate_combinations(self, text):
        """Extract and combine subjects and predicates from different sentences."""
        doc = self.nlp(text)
        subjects = []
        predicates = []
        
        for sent in doc.sents:
            root = None
            subject = None
            
            # Find the root verb and subject
            for token in sent:
                if token.dep_ == 'ROOT':
                    root = token
                if token.dep_ == 'nsubj':
                    subject = token
                    
            if root and subject:
                # Get the subject phrase
                subj_phrase = ' '.join([t.text for t in subject.subtree])
                subjects.append(subj_phrase)
                
                # Get the predicate (everything after subject)
                pred_start = False
                pred_tokens = []
                for token in sent:
                    if token == subject:
                        pred_start = True
                        continue
                    if pred_start:
                        pred_tokens.append(token.text)
                predicates.append(' '.join(pred_tokens))
        
        # Create new combinations
        combinations = []
        for subj in subjects:
            for pred in predicates:
                if subj and pred:
                    combinations.append(f"{subj} {pred}")
                    
        return combinations

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
import random
import spacy
from itertools import combinations
import re

class TextAugmenter:
    def __init__(self):
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('averaged_perceptron_tagger')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        
        self.nlp = spacy.load('en_core_web_sm')
        self.segmenter = TextSegmenter()
        
    def _get_wordnet_pos(self, tag):
        """Map NLTK POS tag to WordNet POS tag"""
        tag_dict = {
            'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ,
            'JJS': wordnet.ADJ,
            'NN': wordnet.NOUN,
            'NNS': wordnet.NOUN,
            'NNP': wordnet.NOUN,
            'NNPS': wordnet.NOUN,
            'RB': wordnet.ADV,
            'RBR': wordnet.ADV,
            'RBS': wordnet.ADV,
            'VB': wordnet.VERB,
            'VBD': wordnet.VERB,
            'VBG': wordnet.VERB,
            'VBN': wordnet.VERB,
            'VBP': wordnet.VERB,
            'VBZ': wordnet.VERB
        }
        return tag_dict.get(tag, None)

    def _get_synonyms(self, word, pos=None):
        """Get synonyms for a word with optional POS"""
        synonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym.split()) == 1:  # Only single words
                    synonyms.add(synonym)
        return list(synonyms)

    def synonym_replacement(self, text, replacement_ratio=0.5):  # Increased from 0.3 to 0.5
        """Replace words with their synonyms"""
        words = word_tokenize(text)
        tagged = pos_tag(words)
        new_words = words.copy()
        
        # Get replaceable word indices (words with valid POS tags)
        replaceable = [
            i for i, (word, tag) in enumerate(tagged)
            if self._get_wordnet_pos(tag) and not word.lower() in ['is', 'are', 'was', 'were']
        ]
        
        # Randomly select words to replace
        num_to_replace = max(2, int(len(replaceable) * replacement_ratio))  # Ensure at least 2 words are replaced
        replace_indices = random.sample(replaceable, min(num_to_replace, len(replaceable)))
        
        for idx in replace_indices:
            word = words[idx]
            tag = tagged[idx][1]
            pos = self._get_wordnet_pos(tag)
            synonyms = self._get_synonyms(word, pos)
            
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)

    def random_deletion(self, text, p=0.2):  # Increased from 0.1 to 0.2
        """Randomly delete words from the text"""
        words = word_tokenize(text)
        # Don't delete if text is too short
        if len(words) <= 3:
            return text
            
        # Create list of words to keep
        new_words = []
        for word in words:
            # Don't delete punctuation or important words
            if word in ['.', ',', '!', '?'] or word.lower() in ['is', 'are', 'was', 'were']:
                new_words.append(word)
            elif random.random() > p:
                new_words.append(word)
                
        # Make sure we didn't delete too much (reduced minimum from 3 to 2)
        if len(new_words) < 2:
            return text
            
        return ' '.join(new_words)

    def random_swap(self, text, n=2):  # Increased from 1 to 2 swaps
        """Randomly swap n pairs of words in the text"""
        words = word_tokenize(text)
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 4:  # Only swap if text is long enough
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                # Don't swap punctuation
                if not (new_words[idx1] in ['.', ',', '!', '?'] or 
                    new_words[idx2] in ['.', ',', '!', '?']):
                    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)

    def _get_random_segment_combination(self, text) -> str:
        """Randomly selects and combines different types of segments"""
        segments = []
        
        # Get all possible segments
        full_sentences = self.segmenter.get_full_sentences(text)
        cross_segments = self.segmenter.get_cross_sentence_segments(text)
        subj_pred_combinations = self.segmenter.get_subject_predicate_combinations(text)
        
        # Increased likelihood of using multiple segments
        num_segments = random.randint(2, 3)  # Changed from always using 2 segments
        
        # Get available segment types
        segment_types = [
            cross_segments,
            subj_pred_combinations,
            full_sentences
        ]
        segment_types = [seg_type for seg_type in segment_types if seg_type]
        
        # Select segments
        for _ in range(num_segments):
            if segment_types:
                segment_list = random.choice(segment_types)
                if segment_list:
                    segments.append(random.choice(segment_list))
            
        if not segments and full_sentences:
            segments.append(random.choice(full_sentences))
            
        random.shuffle(segments)
        
        text = ' '.join(segments)
        text = text.replace('..', '.')
        text = text.replace('. .', '.')
        
        return text

    def generate_augmentations(self, text: str, num_augmentations: int = 2):
        """Generate specified number of augmented versions of the text using multiple strategies"""
        if not text or len(text.strip()) == 0:
            return []

        # Increased likelihood of combining methods
        augmentation_methods = [
            lambda t: self._get_random_segment_combination(t),
            lambda t: self.synonym_replacement(t),
            lambda t: self.random_deletion(t),
            lambda t: self.random_swap(t),
            # More aggressive combinations
            lambda t: self.synonym_replacement(self.random_deletion(t)),
            lambda t: self.random_swap(self._get_random_segment_combination(t)),
            lambda t: self.synonym_replacement(self.random_swap(t)),
            lambda t: self.random_deletion(self._get_random_segment_combination(t))
        ]

        augmentations = []
        for _ in range(num_augmentations):
            # Apply multiple augmentations with higher probability
            if random.random() < 0.7:  # 70% chance to combine methods
                method1 = random.choice(augmentation_methods)
                method2 = random.choice(augmentation_methods)
                augmented_text = method2(method1(text))
            else:
                method = random.choice(augmentation_methods)
                augmented_text = method(text)
            
            # Basic cleanup
            augmented_text = re.sub(r'\s+', ' ', augmented_text).strip()
            augmented_text = re.sub(r'\s+([.,!?])', r'\1', augmented_text)
            
            if augmented_text and augmented_text != text:
                augmentations.append(augmented_text)
            
        if len(augmentations) < num_augmentations:
            return None
        # # If we didn't get enough valid augmentations, try again with simpler methods
        # while len(augmentations) < num_augmentations:
        #     augmented_text = self.synonym_replacement(text, replacement_ratio=0.3)
        #     if augmented_text and augmented_text != text:
        #         augmentations.append(augmented_text)

        return augmentations[:num_augmentations]


import nltk
import random
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
class HNMTextAugmenter():
    def __init__(self):
        # Download necessary NLTK data if not already present
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        import os
        os.environ["MODEL_DIR"] = '../model'
        import nlpaug.augmenter.char as nac
        import nlpaug.augmenter.word as naw

        self.aug0 = nac.OcrAug()
        self.aug1 = nac.KeyboardAug()
        self.aug2 = nac.RandomCharAug(action="insert")
        self.aug3 = nac.RandomCharAug(action="swap")
        self.aug4 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

        self.neg_aug = naw.AntonymAug()

        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('averaged_perceptron_tagger')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        
        self.nlp = spacy.load('en_core_web_sm')
        self.segmenter = TextSegmenter()
        
    def _get_wordnet_pos(self, tag):
        """Map NLTK POS tag to WordNet POS tag"""
        tag_dict = {
            'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ,
            'JJS': wordnet.ADJ,
            'NN': wordnet.NOUN,
            'NNS': wordnet.NOUN,
            'NNP': wordnet.NOUN,
            'NNPS': wordnet.NOUN,
            'RB': wordnet.ADV,
            'RBR': wordnet.ADV,
            'RBS': wordnet.ADV,
            'VB': wordnet.VERB,
            'VBD': wordnet.VERB,
            'VBG': wordnet.VERB,
            'VBN': wordnet.VERB,
            'VBP': wordnet.VERB,
            'VBZ': wordnet.VERB
        }
        return tag_dict.get(tag, None)

    def _get_synonyms(self, word, pos=None):
        """Get synonyms for a word with optional POS"""
        synonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym.split()) == 1:  # Only single words
                    synonyms.add(synonym)
        return list(synonyms)

    def synonym_replacement(self, text, replacement_ratio=0.5):  # Increased from 0.3 to 0.5
        """Replace words with their synonyms"""
        words = word_tokenize(text)
        tagged = pos_tag(words)
        new_words = words.copy()
        
        # Get replaceable word indices (words with valid POS tags)
        replaceable = [
            i for i, (word, tag) in enumerate(tagged)
            if self._get_wordnet_pos(tag) and not word.lower() in ['is', 'are', 'was', 'were']
        ]
        
        # Randomly select words to replace
        num_to_replace = max(2, int(len(replaceable) * replacement_ratio))  # Ensure at least 2 words are replaced
        replace_indices = random.sample(replaceable, min(num_to_replace, len(replaceable)))
        
        for idx in replace_indices:
            word = words[idx]
            tag = tagged[idx][1]
            pos = self._get_wordnet_pos(tag)
            synonyms = self._get_synonyms(word, pos)
            
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)

    def random_deletion(self, text, p=0.2):  # Increased from 0.1 to 0.2
        """Randomly delete words from the text"""
        words = word_tokenize(text)
        # Don't delete if text is too short
        if len(words) <= 3:
            return text
            
        # Create list of words to keep
        new_words = []
        for word in words:
            # Don't delete punctuation or important words
            if word in ['.', ',', '!', '?'] or word.lower() in ['is', 'are', 'was', 'were']:
                new_words.append(word)
            elif random.random() > p:
                new_words.append(word)
                
        # Make sure we didn't delete too much (reduced minimum from 3 to 2)
        if len(new_words) < 2:
            return text
            
        return ' '.join(new_words)

    def random_swap(self, text, n=2):  # Increased from 1 to 2 swaps
        """Randomly swap n pairs of words in the text"""
        words = word_tokenize(text)
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 4:  # Only swap if text is long enough
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                # Don't swap punctuation
                if not (new_words[idx1] in ['.', ',', '!', '?'] or 
                    new_words[idx2] in ['.', ',', '!', '?']):
                    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)

    def _get_random_segment_combination(self, text) -> str:
        """Randomly selects and combines different types of segments"""
        segments = []
        
        # Get all possible segments
        full_sentences = self.segmenter.get_full_sentences(text)
        cross_segments = self.segmenter.get_cross_sentence_segments(text)
        subj_pred_combinations = self.segmenter.get_subject_predicate_combinations(text)
        
        # Increased likelihood of using multiple segments
        num_segments = random.randint(2, 3)  # Changed from always using 2 segments
        
        # Get available segment types
        segment_types = [
            cross_segments,
            subj_pred_combinations,
            full_sentences
        ]
        segment_types = [seg_type for seg_type in segment_types if seg_type]
        
        # Select segments
        for _ in range(num_segments):
            if segment_types:
                segment_list = random.choice(segment_types)
                if segment_list:
                    segments.append(random.choice(segment_list))
            
        if not segments and full_sentences:
            segments.append(random.choice(full_sentences))
            
        random.shuffle(segments)
        
        text = ' '.join(segments)
        text = text.replace('..', '.')
        text = text.replace('. .', '.')
        
        return text
    
    def swap_subject_and_pp_object(self, sentences):
        new_sentences = []
        for sentence in sentences.split('.'):
            doc = self.nlp(sentence.strip())
            subject = None
            pp_object = None
            # Find the subject and the object of the preposition "in"
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    # Get the full phrase for the subject (including determiners)
                    subject_tokens = list(token.subtree)
                    subject = " ".join(t.text for t in subject_tokens)
                if token.dep_ == "pobj" and token.head.text.lower() == "in":
                    # Get the full phrase for the object of the preposition
                    pobj_tokens = list(token.subtree)
                    pp_object = " ".join(t.text for t in pobj_tokens)
            
            # If both components are found, swap their positions
            # if subject and pp_object:
            new_sentences.append(f"{pp_object} is in {subject}")
        
                # continue
        return ('. '.join(new_sentences)).strip()

    def get_synonym(self, word):
        """
        Given a noun, return a random synonym from WordNet if one exists,
        otherwise return the original word.
        """
        synsets = wn.synsets(word, pos=wn.NOUN)
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                # Avoid returning the same word and replace underscores with spaces
                if lemma.name().lower() != word.lower():
                    synonyms.add(lemma.name().replace('_', ' '))
        if synonyms:
            return random.choice(list(synonyms))
        else:
            return word
        
    def random_deletion(self, text, p=0.2):  # Increased from 0.1 to 0.2
        """Randomly delete words from the text"""
        words = word_tokenize(text)
        # Don't delete if text is too short
        if len(words) <= 3:
            return text
            
        # Create list of words to keep
        new_words = []
        for word in words:
            # Don't delete punctuation or important words
            if word in ['.', ',', '!', '?'] or word.lower() in ['is', 'are', 'was', 'were']:
                new_words.append(word)
            elif random.random() > p:
                new_words.append(word)
                
        # Make sure we didn't delete too much (reduced minimum from 3 to 2)
        if len(new_words) < 2:
            return text
        
        return ' '.join(new_words)

    def replace_nouns_with_synonyms(self, sentence):
        """
        Tokenize the sentence, tag each token, and replace any noun with one of its synonyms.
        """
        words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        new_words = []
        for word, tag in tagged_words:
            # Check if the part-of-speech tag indicates a noun (NN, NNS, NNP, NNPS)
            if tag in ("NN", "NNS", "NNP", "NNPS"):
                new_word = self.get_synonym(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        # Reconstruct the sentence and adjust spacing around punctuation
        new_sentence = ' '.join(new_words)
        new_sentence = new_sentence.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        return new_sentence
    
    def get_related_noun(self, word):
        """
        Given a noun, return a random related noun from WordNet that is close in meaning but not exactly the same.
        The method uses the hypernym–hyponym structure:
        1. Retrieve synsets for the word (as a noun).
        2. For each synset, get its hypernyms.
        3. From each hypernym, retrieve its hyponyms (which are "cousins" of the original word).
        4. Filter out candidates that are the same as the original word.
        If no candidate is found, the original word is returned.
        """
        
        synsets = wn.synsets(word, pos=wn.NOUN)
        related = set()
        for syn in synsets:
            hypernyms = syn.hypernyms()
            for hyper in hypernyms:
                for hyponym in hyper.hyponyms():
                    for lemma in hyponym.lemmas():
                        candidate = lemma.name().replace('_', ' ')
                        if candidate.lower() != word.lower():
                            related.add(candidate)
        if related:
            return random.choice(list(related))
        else:
            return word

    def replace_nouns_with_related(self, sentence):
        """
        Tokenize the input sentence, tag its parts of speech, and replace each noun with a related noun.
        The replacement is based on the get_related_noun function.
        """
        words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        new_words = []
        for word, tag in tagged_words:
            if tag in ("NN", "NNS", "NNP", "NNPS"):
                new_word = self.get_related_noun(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        # Reconstruct the sentence and correct spacing around punctuation
        new_sentence = ' '.join(new_words)
        new_sentence = new_sentence.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        return new_sentence
    
    def generate_augmentations(self, text: str, only_neg=False):
        """Generate specified number of augmented versions of the text using multiple strategies"""
        if not only_neg:
            pos = self.replace_nouns_with_synonyms(text)
            if random.random() < 0.05:
                pos = self.aug0.augment(pos)[0]
            if random.random() < 0.05:
                pos = self.aug1.augment(pos)[0]
            if random.random() < 0.05:
                pos = self.aug2.augment(pos)[0]
            if random.random() < 0.05:
                pos = self.aug3.augment(pos)[0]

            augmentation_methods = [
                lambda t: self._get_random_segment_combination(t),
                lambda t: self.random_deletion(t),
            ]
            if random.random() < 0.7:  # 70% chance to combine methods
                method1 = random.choice(augmentation_methods)
                method2 = random.choice(augmentation_methods)
                pos = method2(method1(pos))
            else:
                method = random.choice(augmentation_methods)
                pos = method(pos)
            # Basic cleanup
            pos = re.sub(r'\s+', ' ', pos).strip()
            pos = re.sub(r'\s+([.,!?])', r'\1', pos)
            
        neg = text
        if random.random() < 0.3:
            neg = self.neg_aug.augment(neg)[0]
        else:
            neg = self.replace_nouns_with_related(neg)
        if random.random() < 0.05:
            neg = self.aug0.augment(neg)[0]
        if random.random() < 0.05:
            neg = self.aug1.augment(neg)[0]
        if random.random() < 0.05:
            neg = self.aug2.augment(neg)[0]
        if random.random() < 0.05:
            neg = self.aug3.augment(neg)[0]

        augmentation_methods = [
            lambda t: self._get_random_segment_combination(t),
            lambda t: self.random_deletion(t),
            lambda t: self.random_swap(t),
            # More aggressive combinations
            lambda t: self.random_swap(self._get_random_segment_combination(t)),
            lambda t: self.synonym_replacement(self.random_swap(t)),
            lambda t: self.random_deletion(self._get_random_segment_combination(t))
        ]
        for i in range(2):
            neg = random.choice(augmentation_methods)(neg)
        # else:
        #     method = random.choice(augmentation_methods)
        #     neg = method(neg)
        # Basic cleanup
        neg = re.sub(r'\s+', ' ', neg).strip()
        neg = re.sub(r'\s+([.,!?])', r'\1', neg)
        neg_0 = neg
        # self.aug4.augment(neg)

        if not only_neg:
            neg = text
            if random.random() < 0.3:
                neg = self.neg_aug.augment(neg)[0]
            else:
                neg = self.replace_nouns_with_related(neg)
            # if random.random() < 0.3:
            #     neg = self.swap_subject_and_pp_object(neg)
            if random.random() < 0.1:
                neg = self.aug0.augment(neg)[0]
            if random.random() < 0.1:
                neg = self.aug1.augment(neg)[0]
            if random.random() < 0.1:
                neg = self.aug2.augment(neg)[0]
            if random.random() < 0.1:
                neg = self.aug3.augment(neg)[0]
            augmentation_methods = [
                lambda t: self._get_random_segment_combination(t),
                lambda t: self.random_deletion(t),
                lambda t: self.random_swap(t),
                # More aggressive combinations
                lambda t: self.random_swap(self._get_random_segment_combination(t)),
                lambda t: self.synonym_replacement(self.random_swap(t)),
                lambda t: self.random_deletion(self._get_random_segment_combination(t))
            ]
            for i in range(2):
                neg = random.choice(augmentation_methods)(neg)
            # Basic cleanup
            neg = re.sub(r'\s+', ' ', neg).strip()
            neg = re.sub(r'\s+([.,!?])', r'\1', neg)
            neg_1 = neg
            # self.aug4.augment(neg)
        if only_neg:
            return text, neg_0
        else:
            return text, neg_0, pos, neg_1
        
# The negative augmentation class (jigsaw and cutmix) extending the DINO augmentations.
# Copy of the HNMImageAugmenter class with the new "cutpaste" augmentation.
class HNMImageAugmenter(object):
    def __init__(
        self,
        global_crops_scale=[0.32, 1.0],
        local_crops_scale=[0.05, 0.32],
        local_crops_number=0,
        global_crops_size=224,
        local_crops_size=224,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        negative_prob=0.5,  # probability threshold for negative augmentation method selection
        args=None,
    ):
        if args is not None:
            image_mean = args.image_mean
            image_std = args.image_std

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.negative_prob = negative_prob

        logging.info("###################################")
        logging.info("Using data augmentation parameters:")
        logging.info(f"global_crops_scale: {global_crops_scale}")
        logging.info(f"local_crops_scale: {local_crops_scale}")
        logging.info(f"local_crops_number: {local_crops_number}")
        logging.info(f"global_crops_size: {global_crops_size}")
        logging.info(f"local_crops_size: {local_crops_size}")
        logging.info(f"negative augmentation probability: {negative_prob}")
        logging.info("###################################")

        self.stable_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size, scale=[0.5, 1.0], interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # Geometric augmentations.
        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # Color distortions and blurring.
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
        ])
        local_transfo_extra = GaussianBlur(p=0.5)

        normalize = make_normalize_transform(image_mean, image_std)
        self.normalize = transforms.Compose([
            transforms.Resize((global_crops_size, global_crops_size)),
            transforms.ToTensor(),
            normalize,
        ])
        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def jigsaw(self, image, min_patches=4, max_patches=9, height_range=(2, 3), width_range=(2, 3)):
        possible_configs = []
        for num_h in range(height_range[0], height_range[1] + 1):
            for num_w in range(width_range[0], width_range[1] + 1):
                if num_h * num_w >= min_patches and num_h * num_w <= max_patches:
                    possible_configs.append((num_h, num_w))
        num_h, num_w = random.choice(possible_configs)
        width, height = image.size
        patch_width = width // num_w
        patch_height = height // num_h
        patches = []
        for i in range(num_h):
            for j in range(num_w):
                left = j * patch_width
                upper = i * patch_height
                patch = image.crop((left, upper, left + patch_width, upper + patch_height))
                patches.append(patch)
        random.shuffle(patches)
        new_image = Image.new('RGB', (patch_width * num_w, patch_height * num_h))
        idx = 0
        for i in range(num_h):
            for j in range(num_w):
                new_image.paste(patches[idx], (j * patch_width, i * patch_height))
                idx += 1
        return new_image

    def cutmix(self, image1, image2, blend_ratio=None):
        if blend_ratio is None:
            blend_ratio = random.uniform(0.4, 0.5)
        scale = random.uniform(0.1, 0.4)
        new_width = int(image1.width * scale)
        new_height = int(image1.height * scale)
        image2_resized = image2.resize((new_width, new_height))
        if scale < 1:
            background = Image.new('RGB', image1.size)
            max_x = image1.width - new_width
            max_y = image1.height - new_height
            offset_x = random.randint(0, max_x)
            offset_y = random.randint(0, max_y)
            background.paste(image2_resized, (offset_x, offset_y))
            image2_final = background
        elif scale > 1:
            max_x = new_width - image1.width
            max_y = new_height - image1.height
            offset_x = random.randint(0, max_x)
            offset_y = random.randint(0, max_y)
            image2_final = image2_resized.crop((offset_x, offset_y, offset_x + image1.width, offset_y + image1.height))
        else:
            image2_final = image2_resized
        blended_image = Image.blend(image1, image2_final, blend_ratio)
        return blended_image

    def cutout(self, image, mask_size_ratio_range=(0.4, 0.7), fill_value=127):
        width, height = image.size
        mask_size_ratio = random.uniform(*mask_size_ratio_range)
        mask_width = int(width * mask_size_ratio)
        mask_height = int(height * mask_size_ratio)
        x = random.randint(0, max(width - mask_width, 0))
        y = random.randint(0, max(height - mask_height, 0))
        image_np = np.array(image)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_np[y:y+mask_height, x:x+mask_width, :] = fill_value
        else:
            image_np[y:y+mask_height, x:x+mask_width] = fill_value
        return Image.fromarray(image_np)

    def cutpaste(self, image0, image1, patch_scale_range=(0.3, 0.5)):
        width, height = image0.size
        # Determine patch size as a fraction of image0 dimensions.
        patch_scale = random.uniform(*patch_scale_range)
        patch_width = int(width * patch_scale)
        patch_height = int(height * patch_scale)

        # Random location in image0 for pasting.
        x0 = random.randint(0, max(width - patch_width, 0))
        y0 = random.randint(0, max(height - patch_height, 0))

        # Ensure image1 is large enough to extract a patch.
        width1, height1 = image1.size
        if width1 < patch_width or height1 < patch_height:
            image1 = image1.resize((max(width1, patch_width), max(height1, patch_height)), resample=Image.BICUBIC)
            width1, height1 = image1.size
            
        # Randomly select a patch from image1.
        x1 = random.randint(0, max(width1 - patch_width, 0))
        y1 = random.randint(0, max(height1 - patch_height, 0))
        patch_from_image1 = image1.crop((x1, y1, x1 + patch_width, y1 + patch_height))

        # Paste the patch directly onto a copy of image0.
        image0_copy = image0.copy()
        image0_copy.paste(patch_from_image1, (x0, y0))
        return image0_copy

    def process_single_image(self, image):
        image = self.stable_augmentation_global(image.convert('RGB'))
        image = self.global_transfo1(image)
        return image
    
    def patch_aug(self, image0, image1):
        neg_method = random.sample(['jigsaw', 'cutmix', 'cutout', 'cutpaste'], 2)
        neg_img = image0.convert('RGB')
        if 'jigsaw' in neg_method:
            neg_img = self.jigsaw(self.stable_augmentation_global(neg_img))
        elif 'cutmix' in neg_method:
            neg_img = self.cutmix(
                self.stable_augmentation_global(neg_img), 
                self.stable_augmentation_global(image1.convert('RGB'))
            )
        elif 'cutout' in neg_method:
            neg_img = self.cutout(self.stable_augmentation_global(neg_img), mask_size_ratio_range=(0.1, 0.3))
        elif 'cutpaste' in neg_method:
            neg_img = self.cutpaste(
                self.stable_augmentation_global(neg_img),
                self.stable_augmentation_global(image1.convert('RGB')),
                patch_scale_range=(0.1, 0.3)
            )
        return neg_img
    
    def __call__(self, image0, image1=None, image2=None, no_neg_aug=False, only_neg=False):
        if image1 is None:
            image1 = image0
        im1_base = self.geometric_augmentation_global(image0.convert('RGB'))
        output_1 = self.global_transfo1(im1_base)
        if only_neg:
            neg_img = self.patch_aug(image1, self.geometric_augmentation_local(image0.convert('RGB')))
            negative_output_0 = self.global_transfo1(neg_img)
        else:
            rand = random.random()
            if rand < 0.7:
                im2_base = self.geometric_augmentation_global(image0.convert('RGB'))
                output_2 = self.global_transfo2(im2_base)
            elif rand < 0.9:
                output_2 = self.patch_aug(image0, image1)
                output_2 = self.global_transfo2(output_2)
            else:
                output_2 = self.local_transfo(self.geometric_augmentation_local(image0.convert('RGB')))

            neg_img = self.patch_aug(image2, self.geometric_augmentation_local(image0.convert('RGB')))
            negative_output_0 = self.global_transfo1(neg_img)

            neg_img = self.patch_aug(image2, self.geometric_augmentation_local(image1.convert('RGB')))
            negative_output_1 = self.global_transfo1(neg_img)

        if only_neg:
            return output_1, negative_output_0
        else:
            return output_1, negative_output_0, output_2, negative_output_1

class ParquetDataset(IterableDataset):
    def __init__(self, directory_path, tokenizer, args):
        self.directory_path = directory_path
        self.tokenize = tokenizer
        self.file_paths =sorted([
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith('.parquet')
        ])
        random.shuffle(self.file_paths)
        self.file_idx = 0
        self.file = None
        self.text_augmentor = HNMTextAugmenter()

    def _open_next_file(self):
        self.file = pq.ParquetFile(self.file_paths[self.file_idx])
        self.file_idx += 1

    def __iter__(self):
        while self.file_idx < len(self.file_paths):
            self._open_next_file()
            for batch in self.file.iter_batches(batch_size=1):
                df = batch.to_pandas()
                row_data = df.iloc[0]
                # # Process each row in the batch
                # for _, row in df.iterrows():
                text = row_data['caption']
                outputs = self.text_augmentor.generate_augmentations(text)
                if outputs == None:
                    continue
                text0, neg_0, text1, neg_1 = outputs
                text0 = self.tokenize([str(text0)])[0]
                text1 = self.tokenize([str(text1)])[0]
                neg_0 = self.tokenize([str(neg_0)])[0]
                neg_1 = self.tokenize([str(neg_1)])[0]
                yield text0, neg_0, text1, neg_1

    def __len__(self):
        return 100000000

class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)

def make_normalize_transform(
    mean  = OPENAI_DATASET_MEAN,
    std = OPENAI_DATASET_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

def color_jitter(brightness, contrast, saturation, hue, p=0.5):
    jitter_transform = ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    return RandomApply([jitter_transform], p=p)


def gray_scale(p=0.1):
    return RandomGrayscale(p=p)


def _convert_to_rgb(image):
    return image.convert('RGB')


def create_train_transform(image_size, aug_cfg, args):
    if args.image_mean == None or args.image_mean == None:
        normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    else:
        normalize = Normalize(mean=args.image_mean, std=args.image_std)
    train_transform = [
        RandomResizedCrop(
            image_size,
            scale=aug_cfg.get('scale', (0.08, 1.0)),
            interpolation=InterpolationMode.BICUBIC,
        ),
        _convert_to_rgb,
    ]

    if aug_cfg.get('color_jitter_prob', 0) > 0:
        train_transform.append(
            color_jitter(*aug_cfg['color_jitter'],
                         p=aug_cfg['color_jitter_prob'])
        )

    if aug_cfg.get('gray_scale_prob', 0) > 0:
        train_transform.append(
            gray_scale(aug_cfg['gray_scale_prob'])
        )

    train_transform.extend([
        ToTensor(),
        normalize
    ])
    return Compose(train_transform)


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale=[0.32, 1.0],
        local_crops_scale=[0.05, 0.32],
        local_crops_number=0,
        global_crops_size=224,
        local_crops_size=224,
        image_mean=[0.5,0.5,0.5],
        image_std=[0.5,0.5,0.5],
        args=None,
    ):
        if args is not None:
            image_mean = args.image_mean
            image_std = args.image_std
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logging.info("###################################")
        logging.info("Using data augmentation parameters:")
        logging.info(f"global_crops_scale: {global_crops_scale}")
        logging.info(f"local_crops_scale: {local_crops_scale}")
        logging.info(f"local_crops_number: {local_crops_number}")
        logging.info(f"global_crops_size: {global_crops_size}")
        logging.info(f"local_crops_size: {local_crops_size}")
        logging.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )
        local_transfo_extra = GaussianBlur(p=0.5)

        normalize = make_normalize_transform(image_mean, image_std)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image0, image1=None):
        # global crops:
        if image1 is None:
            image1 = image0
        im1_base = self.geometric_augmentation_global(image0.convert('RGB'))
        output_1 = self.global_transfo1(im1_base)

        if random.random() < 0.9:
            im2_base = self.geometric_augmentation_global(image1.convert('RGB'))
            output_2 = self.global_transfo2(im2_base)
        else:
            output_2 = self.local_transfo(self.geometric_augmentation_local(image1.convert('RGB'))) 

        return output_1, output_2

class PlainImageDataset(Dataset):
    def __init__(self, input_filename, args):
        logging.debug(f'Loading plain data from {input_filename}.')
        self.input_filename = input_filename
        self.images = os.listdir(input_filename)
        self.aug_transform = DataAugmentationDINO(
            global_crops_size=args.force_image_size, 
            local_crops_size=args.force_image_size, 
            args=args
        )
        # self.aug_transform = create_train_transform(
        #     args.force_image_size, args.aug_cfg, args)
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            path = os.path.join(self.input_filename, self.images[idx])
            if path.endswith('tif'):
                with Image.open(path) as img:
                    img.seek(0)
            else:
                img = Image.open(path)
            image_0, image_1 = self.aug_transform(img)
            return image_0, image_1
        except Exception as e:
            logging.warning(f"Error processing index {idx}: {e}")
            return self.__getitem__((idx + 1) % self.__len__())

class CsvImageDataset(Dataset):
    def __init__(self, input_filename, transforms, args, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.images = df['images'].tolist()
        self.is_bytes = isinstance(eval(self.images[0])[0], bytes)
        self.aug_transform = DataAugmentationDINO(
            global_crops_size=args.force_image_size, 
            local_crops_size=args.force_image_size, 
            args=args
        )
        # self.aug_transform = create_train_transform(
        #     args.force_image_size, args.aug_cfg, args)
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            images = eval(self.images[idx])
            # print(images)
            if not self.is_bytes:
                if len(images) > 1:
                    idx0 = random.randint(0, len(images)-1)
                    use_same = random.random() < 0.5  # 50% chance for same index
                    if use_same:
                        idx1 = idx0
                    else:
                        if idx0 == 0:  # If at start, can only go forward
                            idx1 = idx0 + 1
                        elif idx0 == len(images)-1:  # If at end, can only go backward
                            idx1 = idx0 - 1
                        else:  # Otherwise randomly choose between forward or backward
                            idx1 = idx0 + random.choice([-1, 1])
                    sample0 = images[idx0]
                    sample1 = images[idx1]
                    # image_0 = self.aug_transform(Image.open(str(sample0)))
                    # image_1 = self.aug_transform(Image.open(str(sample1)))
                    image_0, image_1 = self.aug_transform(Image.open(str(sample0)), Image.open(str(sample1)))
                    return image_0, image_1
                else:
                    image = Image.open(str(images[0]))
                    # image_0 = self.aug_transform(image)
                    # image_1 = self.aug_transform(image)
                    image_0, image_1 = self.aug_transform(image, image)
                    return image_0, image_1
            else:
                if len(images) > 1:
                    idx0 = random.randint(0, len(images)-1)
                    use_same = random.random() < 0.5  # 50% chance for same index
                    if use_same:
                        idx1 = idx0
                    else:
                        if idx0 == 0:  # If at start, can only go forward
                            idx1 = idx0 + 1
                        elif idx0 == len(images)-1:  # If at end, can only go backward
                            idx1 = idx0 - 1
                        else:  # Otherwise randomly choose between forward or backward
                            idx1 = idx0 + random.choice([-1, 1])
                    sample0 = images[idx0]
                    sample1 = images[idx1]
                    # image_0 = self.aug_transform(decode_image_bytes(sample0))
                    # image_1 = self.aug_transform(decode_image_bytes(sample1))
                    image_0, image_1 = self.aug_transform(decode_image_bytes(sample0), decode_image_bytes(sample1))
                    return image_0, image_1
                else:
                    image = decode_image_bytes(images[0])
                    # image_0 = self.aug_transform(image)
                    # image_1 = self.aug_transform(image)
                    image_0, image_1 = self.aug_transform(image, image)
                    return image_0, image_1
        except Exception as e:
            # logging.warning(f"Error processing index {idx}: {e}")
            return self.__getitem__((idx + 1) % self.__len__())

import tarfile
class ImageParquetDataset(IterableDataset):
    def __init__(self, directory_path, transforms, args):
        self.directory_path = directory_path
        self.aug_transform = HNMImageAugmenter(
            global_crops_size=args.force_image_size, 
            local_crops_size=args.force_image_size, 
            args=args
            )
        self.file_paths = sorted([
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith('.tar')
        ])
        self.file_idx = 0
        self.file = None
    
    def __len__(self):
        return 100000000

    def _open_next_file(self):
        self.file_paths = sorted([
            os.path.join(self.directory_path, f)
            for f in os.listdir(self.directory_path)
            if f.endswith('.tar')
        ])
        random.shuffle(self.file_paths)
        self.file = self.file_paths[self.file_idx]
        self.file_idx += 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process case
            file_subsets = self.file_paths
        else:  # Split files across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_subsets = self.file_paths[worker_id::num_workers]

        last_image = None
        previous_last_image = None
        for file_path in file_subsets:
            with tarfile.open(file_path, 'r') as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith(('.jpg', '.jpeg', '.png')):
                        image = self._tar_to_image(tar, member)
                        if previous_last_image is None:
                            previous_last_image = image.copy()
                            continue
                        if last_image is None:
                            last_image = image.copy()
                            continue
                        if image is not None:
                            image_0, neg_image_0, image_1, neg_image1 = self.aug_transform(image, last_image, previous_last_image)
                            previous_last_image = last_image.copy()
                            last_image = image.copy()
                            
                            yield image_0, neg_image_0, image_1, neg_image1

    def _tar_to_image(self, tar, member):
        try:
            image = Image.open(tar.extractfile(member)).convert('RGB')
            return image
        except IOError as e:
            print(f"Error loading image from TAR file {member.name}: {e}")
            return None



import tarfile

class CsvImageTextDataset(IterableDataset):
    def __init__(self, directory_path, transforms=None, tokenizer=None):
        self.directory_path = directory_path
        self.transforms = transforms
        self.tokenize = tokenizer
        self.file_paths = sorted([
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith('.parquet')
        ])
        self.file_idx = 0
        self.file = None

    def _open_next_file(self):
        if self.file:
            self.file.close()
        self.file = pq.ParquetFile(self.file_paths[self.file_idx])
        self.file_idx += 1

    def __iter__(self):
        # Iterate over files
        while self.file_idx < len(self.file_paths):
            self._open_next_file()
            # Iterate over rows in the current file
            # One sample at a time
            for row in self.file.iter_batches(batch_size=1):
                df = row.to_pandas()
                row_data = df.iloc[0]
                url = row_data['url']
                text = row_data['re_caption']
                image = self._url_to_image(url)
                text = self.tokenize([str(text)])[0]
                if image is not None:
                    if self.transforms:
                        image = self.transforms(image)
                    yield image, text

    def _url_to_image(self, url):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            return image
        except (requests.RequestException, IOError) as e:
            print(f"Error loading image from {url}: {e}")
            return None

    # def __del__(self):
    #     if self.file:
    #         self.file.close()


import os
import random
import tarfile
import torch
from PIL import Image
from collections import defaultdict

class ImageTextParquetDataset(torch.utils.data.IterableDataset):
    def __init__(self, directory_path=".", tokenizer=None, args=None):
        super().__init__()
        self.directory_path = '/datasets/mvimagenet/current/augmentation_dataset/imagetext'
        if not os.path.exists(self.directory_path):
            self.directory_path = directory_path
        self.file_paths = sorted([
            os.path.join(self.directory_path, f)
            for f in os.listdir(self.directory_path)
            if f.endswith('.tar')
        ])
        self.aug_transform = HNMImageAugmenter(
            global_crops_size=args.force_image_size, 
            local_crops_size=args.force_image_size, 
            args=args
        )
        self.tokenizer = tokenizer


    def __len__(self):
        # Large “synthetic” length; you can adjust as needed
        return 100000000

    def __iter__(self):
        """
        Iteration logic that:
          1) Splits the .tar files among multiple workers (if applicable).
          2) For each .tar file, groups members (image, caption, neg_image, neg_caption) by prefix.
          3) Extracts each group on-the-fly and yields them.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process case
            file_subsets = self.file_paths
        else:
            # Multi-worker case; distribute tar files among the workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_subsets = self.file_paths[worker_id::num_workers]

        # Shuffle the subset of files if desired
        random.shuffle(file_subsets)

        for file_path in file_subsets:
            try:
                with tarfile.open(file_path, 'r') as tar:
                    group_map = defaultdict(dict)
                    for member in tar:
                        if not member.isfile():
                            continue

                        # Example: "004580013.caption" => prefix="004580013", suffix="caption"
                        parts = member.name.split('.', maxsplit=1)
                        if len(parts) < 2:
                            continue

                        prefix, suffix = parts[0], parts[1]
                        group_map[prefix][suffix] = member

                        if (
                            "image" in group_map[prefix]
                            and "caption" in group_map[prefix]
                            and "neg_image" in group_map[prefix]
                            and "neg_caption" in group_map[prefix]
                        ):
                            anchor_image = self._tar_to_image(tar, group_map[prefix]["image"])
                            anchor_caption = self._tar_to_text(tar, group_map[prefix]["caption"])
                            neg_image = self._tar_to_image(tar, group_map[prefix]["neg_image"])
                            neg_caption = self._tar_to_text(tar, group_map[prefix]["neg_caption"])
                            try:
                                anchor_image = self.aug_transform.process_single_image(anchor_image)
                                neg_image = self.aug_transform.process_single_image(neg_image)
                                anchor_caption = self.tokenizer(anchor_caption)[0]
                                neg_caption = self.tokenizer(neg_caption)[0]
                            except:
                                logging.info('data error')
                                continue
                            # Skip this group if any critical image failed to load
                            if anchor_image is None or neg_image is None:
                                del group_map[prefix]
                                continue

                            yield anchor_image, neg_image, anchor_caption, neg_caption
                            del group_map[prefix]
            except (tarfile.TarError, OSError) as e:
                # Optionally log the error:
                # print(f"Skipping {file_path} due to error: {e}")
                continue

    def _tar_to_image(self, tar, member):
        """Extract an image from the tar member and convert to PIL."""
        try:
            file_obj = tar.extractfile(member)
            if file_obj is None:
                return None
            with Image.open(file_obj) as img:
                return img.convert('RGB')
        except Exception as e:
            print(f"[WARN] Error loading image from {member.name}: {e}")
            return None

    def _tar_to_text(self, tar, member):
        """Extract text caption from the tar member."""
        try:
            file_obj = tar.extractfile(member)
            if file_obj is None:
                return None
            return file_obj.read().decode('utf-8', errors='replace').strip()
        except Exception as e:
            print(f"[WARN] Error loading text from {member.name}: {e}")
            return None

        
class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    # Use default_factory for mutable default
    # shared_epochs: List[SharedEpoch] = field(default_factory=list)
    samplers: List[DistributedSampler] = field(default_factory=list)
    
    def set_epoch(self, epoch):
        # for i in range(len(self.shared_epochs)):
        #     if self.shared_epochs[i] is not None:
        #         self.shared_epochs[i].set_value(epoch)
        if len(self.samplers) > 0 and isinstance(self.samplers[0], DistributedSampler):
            for i in range(len(self.samplers)):
                if self.samplers[i] is not None:
                    self.samplers[i].set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)])
                         for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(
            location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = (
        'png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    try:
        current_sample = None
        for filesample in data:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
            #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
            #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
            if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(
                    __key__=prefix, __url__=filesample["__url__"])
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
    except:
        pass
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])

class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(input_filename, args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, image_only=False):
    input_shards = input_filename
    # input_shards = (
    #     [f"/datasets/laion400m_2022-03-03_1926/images/{i:05d}.tar" for i in range(33500, 40000)] 
    #     + 
    #     [f"/scratch/partial_datasets/datacomp/recap_datacomp_1b_data/{i:05d}.tar" for i in range(0, 7000)]
    # )
    # [f"/datasets/laion400m_2022-03-03_1926/images/{i}.tar" for i in range(33500, 40000)] + \
    # [f"/scratch/partial_datasets/datacomp/recap_datacomp_1b_data/{i}.tar" for i in range(00000, 10000)]
    # n_shards_a = 97001  # {00000..97000}.tar → 97001 shards
    # n_shards_b = 41275  # Adjust if your brace range differs
    # # Weight per shard (so each dataset has total weight = 1)
    # weight_per_shard_a = 1 / n_shards_a
    # weight_per_shard_b = 1 / n_shards_b
    # # Corresponding weights:
    # weights = (
    #     [weight_per_shard_a] * n_shards_a +
    #     [weight_per_shard_b] * n_shards_b
    # )
    # args.train_data_upsampling_factors = weights
    # args.dataset_resampled = True

    # args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    # logging.info(input_shards)
    # raise
    num_shards = None
    args.train_num_samples = 400_000_000
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    # shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            # epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    # epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    if image_only:
        def filter_no_image(sample):
            return "image" in sample
        def augment_two_images(sample, augmentation_fn):
            # "sample" is a dictionary containing at least the key "image"
            img = sample["image"]
            aug1, aug2 = augmentation_fn(img)  # call your DataAugmentationDINO instance
            sample["image1"] = aug1
            sample["image2"] = aug2
            return sample
        augmentation_fn = DataAugmentationDINO(
            global_crops_size=224,
            local_crops_size=224,
        )
        pipeline.extend([
            wds.select(filter_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp"),
            wds.map(lambda sample: augment_two_images(sample, augmentation_fn)),
            wds.to_tuple("image1", "image2"),
            wds.batched(args.batch_size, partial=not is_train)
        ])
    else:
        # logging.info(args.force_image_size)
        # raise
        imageaug = HNMImageAugmenter(
            global_crops_size=args.force_image_size, 
            local_crops_size=args.force_image_size, 
            args=args
        )
        textaug = HNMTextAugmenter()
        # def add_filename(sample):
        #     # This assumes sample has a '__url__' key (or you might use '__key__')
        #     sample["filename"] = sample.get("__url__", sample.get("__key__", None))
        #     return sample

        # Then, in your pipeline definition, add:
        # pipeline.append(wds.map(add_filename))

        def process_batch(batch):
            images, neg_images, texts, neg_texts = [], [], [], []
            # filename = batch.get("filename", [None]*len(batch["image"]))[i]
            last_image = batch[0][-1].copy()
            for i in range(len(batch[0])):
                # Process the image and its negative version.
                image, neg_image = imageaug(batch[0][i], last_image, only_neg=True)
                last_image = batch[0][i].copy()
                data = batch[1][i]
                # Choose between the original caption or an alternative one.
                caption = data["org_caption"].lower() if random.random() > 0.1 else data["caption"].lower()
                # Generate text augmentations and tokenize.
                text, neg_text = textaug.generate_augmentations(caption, only_neg=True)
                text = tokenizer(text)[0]
                neg_text = tokenizer(neg_text)[0]
                # Append results to batch lists.
                images.append(image)
                neg_images.append(neg_image)
                texts.append(text)
                neg_texts.append(neg_text)
            
            return torch.stack(images), torch.stack(neg_images), torch.stack(texts), torch.stack(neg_texts)

            
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.to_tuple("image", "json"),
            wds.batched(args.batch_size, partial=not is_train),
            wds.map(process_batch)
        ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    logging.info(f'======the legnth for webdataset===== for {input_filename}: {num_samples}')
    return dataloader

def get_parquet_text_dataset(input_filename, args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert input_filename
    dataset = ParquetDataset(
        input_filename,
        tokenizer=tokenizer,
        args=args,
    )
    return dataset

def get_parquet_image_dataset(input_filename, args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert input_filename
    dataset = ImageParquetDataset(
        input_filename,
        preprocess_fn,
        args=args,
    )
    return dataset

def get_csv_image_dataset(input_filename, args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert input_filename
    dataset = CsvImageDataset(
        input_filename,
        preprocess_fn,
        args=args,
    )
    return dataset


def get_csv_image_text_dataset(input_filename, args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert input_filename
    dataset = CsvImageTextDataset(
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer
    )
    return dataset

def get_plain_image_dataset(input_filename, args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert input_filename
    dataset = PlainImageDataset(
        input_filename,
        args,
    )
    return dataset

def get_csv_dataset(input_filename, args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(
        dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

from functools import partial
def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "webdataset_image":
        return partial(get_wds_dataset, image_only=True)
    elif dataset_type == "plain_image":
        return get_plain_image_dataset
    elif dataset_type == "parquet_text":
        return get_parquet_text_dataset
    elif dataset_type == "parquet_image":
        return get_parquet_image_dataset
    elif dataset_type == "csv_image":
        return get_csv_image_dataset
    elif dataset_type == "csv_image_text":
        return get_csv_image_text_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        datasets = {}
        all_loaders = []
        all_samplers = []
        # all_shared_epochs = []
        for key in args.train_data:
            if "webdataset" in key:
                paths = args.train_data[key][0]
                # logging.info('===ds===the paths for webdataset=====\n', paths)
                dataloader = get_dataset_fn(paths, key)(
                    paths, args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
                sampler = None
            else:
                if key in "triplet_image_text":
                    datasets[key] = ImageTextParquetDataset(tokenizer=tokenizer, args=args)
                else:
                    datasets[key] = []
                    for path in args.train_data[key]:
                        dst = get_dataset_fn(path, key)(
                            path, args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
                        if isinstance(dst, IterableDataset):
                            datasets[key] = dst
                            break
                        else:
                            datasets[key].append(dst)
                

                if not isinstance(datasets[key], IterableDataset) or isinstance(datasets[key], DataInfo):
                    datasets[key] = ConcatDataset(datasets[key])
                # shared_epoch = None
                is_train = True

                sampler = DistributedSampler(
                    datasets[key]) if args.distributed and is_train else None
                shuffle = is_train and sampler is None
                dataloader = DataLoader(
                    datasets[key],
                    batch_size=args.batch_size,
                    shuffle=shuffle if not isinstance(
                        datasets[key], IterableDataset) else False,
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=sampler if not isinstance(
                        datasets[key], IterableDataset) else None,
                    drop_last=is_train,
                )
                if not isinstance(datasets[key], IterableDataset):
                    num_samples = len(datasets[key])
                    dataloader.num_samples = num_samples
                    dataloader.num_batches = len(dataloader)
            all_loaders.append(dataloader)
            all_samplers.append(sampler)
            # all_shared_epochs.append(shared_epoch)
        combined_dataloader = CombinedDataLoader(
            all_loaders, args.batch_size, accum_freq=args.accum_freq)
        data["train"] = DataInfo(combined_dataloader, all_samplers)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data


class CombinedDataLoader:
    def __init__(self, dataloaders, batch_size=1, accum_freq=128):
        """
        Initialize the CombinedDataLoader with multiple dataloaders.
        Args:
            dataloaders: List of DataLoader objects to combine.
            batch_size: Batch size for the combined dataloader.
            accum_freq: Number of batches to accumulate before switching dataloaders.
        """
        self.dataloaders = dataloaders
        self.iterators = [iter(dl) for dl in dataloaders]
        self.batch_size = batch_size
        self.accum_freq = accum_freq
        self.num_samples = 10000000
        if self.num_samples > 0:
            self.num_batches = math.ceil(self.num_samples / self.batch_size)
        else:
            self.num_batches = float('inf')  # Infinite for iterable datasets

    def __iter__(self):
        return self

    def __next__(self):
        """
        Fetch a batch from each dataloader and return them as a list.
        If one dataloader is exhausted, continue with the others.
        """
        try:
            # Get data from each dataloader
            batch_list = [next(dl_iter) for dl_iter in self.iterators]
            # print(batch_list)
            return batch_list

        except StopIteration:
            # If any dataloader is exhausted, reset them
            self.iterators = [iter(dl) for dl in self.dataloaders]
            raise StopIteration("All dataloaders exhausted")
