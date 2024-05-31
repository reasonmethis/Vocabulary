# Vocabulary Training Ground

## Overview

This project unifies word-based algorithms within a text menu interface and comprises two main components: 

1. Preparation for the **VocabMeThis** web app
2. A minimax engine for the **Balda** game

## Components

### 1. VocabMeThis Preparation

A training ground for the VocabMeThis web app, where we:

- **Collect and Build Dictionaries**: Regular and frequency dictionaries.
- **Measure and Train Vocabulary**: Implemented as a simple console interface.

The core NLP tasks are performed in `builddicts.py`:

- **Crawling Wiktionary**: Fetches word definitions (addressing gaps in NLTK).
- **Custom Lemmatization Strategy**: Corrects NLTK's built-in lemmatize function errors (e.g., identifying 'beses' as a form of 'be').
- **Filtering**: Removes derivative word forms, alternative spellings, and nearly identical words (e.g., fantastical and fantastic).
- **Ranking Words**: Uses NLTK corpora to construct frequency dictionaries and ranks words by difficulty.
- **Linking Words**: Establishes connections between words with similar meanings.

### 2. Balda Game Engine

An implementation of a minimax algorithm with alpha-beta pruning for the Balda game, which has not yet been integrated into the VocabMeThis app. Key features include:

- **Game Rules**:
  - Players alternate naming letters.
  - The player who completes an English word (>=3 letters) loses.
  - Players must name letters that form possible words.

  **Example**: If previous letters are T, A, B, L, a player can't avoid completing the word TABLE by saying a random letter like X. If X is chosen, the other player can challenge: "I don't know any words starting with TABLX, name one and you win, but if you don't know one either, then you lose". The game uses an official Scrabble dictionary to validate words.

- **Algorithm Efficiency**:
  - Unlike chess, minimax search here doesn't experience exponential blow-up.
  - The total number of possible games is limited by the total number of valid words.
  - Optimal strategy computed in time O(T), where T = N * avg(L), N = number of words, L = length of word.
  - Trie-based search through each letter of each word.
  - Alpha-beta pruning enhances speed in practice.
