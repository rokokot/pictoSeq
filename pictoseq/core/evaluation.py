#!/usr/bin/env python3



import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging
import math
import time

# we're gonna be using jiwer for wer, but this is optional

try:
  import jiwer
except ImportError:
  HAS_JIWER = False

def safe_encode(text):
  if not isinstance(text,str):
    return str(text)
  return text


class EvaluationMetrics:

  @staticmethod
  def calculate_bleu(predictions, references):
    def get_ngrams(tokens, n):
    

  def calculate_rouge_l(predictions, references):


  def calculate_wer_jiwer(predictions, references):

  def calculate_wer_base(predictions, references):

class FrenchLinguisticAnalysis:
  def __init__(self):
    self.french_articles = {}
    self.french_pronouns = {}
    self.french_prepositions = {}
    self.french_conjunctions = {}



  def analyze_patterns(self, predictions):    # we predefine some patterns to look for  in the sequences