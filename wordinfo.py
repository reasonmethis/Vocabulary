import nltk
from nltk.corpus import wordnet

posdescr_s = """CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective (large)
JJR	adjective, comparative (larger)
JJS	adjective, superlative (largest)
LS	list market
MD	modal (could, will)
NN	noun, singular (cat, tree)
NNS	noun plural (desks)
NNP	proper noun, singular (sarah)
NNPS	proper noun, plural (indians or americans)
PDT	predeterminer (all, both, half)
POS	possessive ending (parent\ 's)
PRP	personal pronoun (hers, herself, him,himself)
PRP$	possessive pronoun (her, his, mine, my, our)
RB	adverb (occasionally, swiftly)
RBR	adverb, comparative (greater)
RBS	adverb, superlative (biggest)
RP	particle (about)
TO	infinitive marker (to)
UH	interjection (goodbye)
VB	verb (ask)
VBG	verb gerund (judging)
VBD	verb past tense (pleaded)
VBN	verb past participle (reunified)
VBP	verb, present tense not 3rd person singular(wrap)
VBZ	verb, present tense with 3rd person singular (bases)
WDT	wh-determiner (that, what)
WP	wh- pronoun (who)
WRB	wh- adverb (how)"""

posdescr_d = {}
for line in posdescr_s.split('\n'):
    linespl = line.strip('\r\n').split('\t')
    posdescr_d[linespl[0]] = linespl[1]


def get_pos(wrd):
    tag = nltk.pos_tag([wrd])[0][1].upper()
    return tag


'''def mylemmatize(wrd):
	lemset = set()
	if wrd in lemmatize_d:
		for poslemset in lemmatize_d[wrd].values():
			lemset |= poslemset
	else:
		lemset.add(wrd)
	return lemset'''


def meanings(wrd, strictmatch=True, include_lemmanames=True):
    syns = wordnet.synsets(wrd)
    pos = get_pos(wrd)
    posdef = []
    if pos[0] not in 'JNVR':
        if include_lemmanames:
            posdef = [wrd + ': ' + posdescr_d[pos]]
        else:
            posdef = [posdescr_d[pos]]

    if include_lemmanames:
        return posdef + [
            ', '.join(x.lemma_names()) + ': ' + x.definition()
            for x in syns
            if not strictmatch or wrd in x.lemma_names()
        ]
    else:
        return posdef + [
            x.definition() for x in syns if not strictmatch or wrd in x.lemma_names()
        ]


def formatted_meanings(wrd, strictmatch=True):
    return "\n".join(['- ' + x for x in meanings(wrd, strictmatch=strictmatch)])
