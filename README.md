Word-based algorithms unified in a menu-based text interface. Has two main components: preparation for VocabMeThis and a minimax engine for the Balda game.

1. Training ground for the VocabMeThis web app, where I collect and build regular dictionaries and frequency dictionaries, and implement the measure and train vocabulary functionality as a simple console interface.

2. Implementation of an engine for the Balda game, which has not (yet) made it into the VocabMeThis app. The engine uses a minimax algorithm with alpha-beta pruning, with some custom enhancements based on the specifics of the game. 

In the game, players alternate naming letters and the player who is forced to complete an English word (>=3 letters long) loses. A player must always name a letter such that a word is possible. For example, if the previous letters were T, A, B, L, then the player whose turn it is can't avoid completing the word TABLE by simply saying a random letter like X. If the player does say X, the other player can issue a challenge: "I don't know any words starting with TABLX, name one and you win, but if you don't know one either then you lose". The game uses an official Scrabble dictionary to determine if a word is valid or not.

Unlike a game like chess, where the minimax search is exponential in the depth of the search, here there's no exponential blow up, since the total number of possible games is limited by the total number of valid words, so the optimal strategy can be computed in time O(T), where T = N * avg(L), N = number of words, L = length of word. In other words, T is the total number of letters in all the words and the algorithm can just go through each letter of each word and build a trie. Alpha-beta pruning makes things even faster in practice.