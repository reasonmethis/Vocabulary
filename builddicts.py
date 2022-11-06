import json
import re
from os import truncate

import nltk
from nltk.corpus import wordnet

from vocabutils import load_freq_dic, load_lines, makefn_balda
from wordinfo import get_pos


# from PyDictionary import PyDictionary
# pydic = PyDictionary()
# print(pydic.meaning('my')); exit()
def is_word(w):
    return (
        w in words and w[0].islower()
    )  # excludes proper nouns but for example July is
    # technically one


def lem(wrd):
    return lemmer.lemmatize(wrd, get_wordnet_pos(wrd))


# Look into
# http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#WordNetCorpusReader.lemma_count
# http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#WordNetCorpusReader.morphy
def lems(wrd):
    return [lemmer.lemmatize(wrd, pos) for pos in 'nvasr']


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    # pos tags give much more detailed information and unlike wordnet include pronouns, prepositions etc
    # https://www.guru99.com/pos-tagging-chunking-nltk.html
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def add_to_freq_dic(fr_d, toks):
    # print('a',end = '')
    cnt = 0
    for wrd in toks:
        # print(wrd)
        wrd = lem(wrd)
        if not is_word(wrd):
            continue
        cnt += 1
        try:
            # print('g', cnt)
            fr_d[wrd] += 1
        except:
            fr_d[wrd] = 1
            print(wrd, len(fr_d))
        if cnt % 1000 == 0:
            print('*', end='')
            # return


def fr_d_to_l(fr_d):
    fr_l = list(fr_d.items())
    fr_l.sort(key=lambda x: -x[1])
    return fr_l


def save_as_str(fn, obj):
    with open(fn, 'w') as f:
        f.write(str(obj))


def save_fr_d(fn, frd):
    save_as_str(fn, fr_d_to_l(frd))


def gen_freq_dic(fr_d, corpora):
    """Generate frequency dictionary from supplied corpora, add to fr_d, and save"""
    for corpus in corpora:
        print(corpus)

        for file_id in corpus.fileids():
            # print(file_id)
            # with open(file_id, 'r') as f:
            # txt = f.read()
            # txt = corpus.raw(file_id)
            # toks = nltk.word_tokenize(txt)
            toks = corpus.words(file_id)
            # print(toks[:100])
            # print(len(toks))
            # input()
            # print(len(toks))
            add_to_freq_dic(fr_d, toks)
            # break
    fr_l = fr_d_to_l(fr_d)
    # print(fr_l[:20])
    save_as_str(fn_fr_d, fr_l)
    return fr_l


def buildFreqDic(fn):
    corpora = [
        nltk.corpus.brown,
        nltk.corpus.gutenberg,
        nltk.corpus.reuters,
        nltk.corpus.webtext,
    ]
    fr_d = {}
    fr_l = gen_freq_dic(fr_d, corpora)
    save_as_str(fn, fr_l)
    input("finished saving dictionary.")


def build_dic():
    """creates frequency dictionaries of wordnet words satisfying certain criteria
    by using, as a reference, an existing frequency dictionary
    Basically makes files fn_lowercase etc."""
    print("Building dictionary...")
    # for ss in wordnet.all_synsets():
    # from itertools import chain
    # lemmas_in_wordnet = list(chain(*[ss.lemma_names() for ss in wordnet.all_synsets()]))
    synsets_in_wordnet = wordnet.all_synsets()
    n = 117658
    frac = 10
    all_d = {}  # all words and expressions with definitions
    onewrd_d = {}
    lowercase_d = {}
    for i, s in enumerate(synsets_in_wordnet):
        if i > n * frac / 100:
            print(f"{frac}% done.")
            frac += 10

        for wrd in s.lemma_names():
            all_d[wrd] = 0
            if '_' in wrd:
                continue
            onewrd_d[wrd] = 0
            if not wrd[0].islower():
                continue
            lowercase_d[wrd] = 0

    print("Building frequency dictionary...")

    fr_l, fr_d = load_freq_dic(fn_fr_d)

    for wrd, freq in fr_d.items():
        # check for words like pronouns that are legitimate words but not part of wordnet
        # like my, she, and add them
        ok = False
        if wrd not in all_d:
            pos = get_pos(wrd)
            if pos in [
                'CC',
                'DT',
                'EX',
                'IN',
                'MD',
                'PDT',
                'PRP',
                'PRP$',
                'RP',
                'TO',
                'UH',
                'WDT',
                'WP',
                'WRB',
            ]:
                ok = True

        if ok or wrd in all_d:
            all_d[wrd] = freq
        if ok or wrd in onewrd_d:
            onewrd_d[wrd] = freq
        if ok or wrd in lowercase_d:
            lowercase_d[wrd] = freq

    print("Saving dictionary...")
    save_fr_d(fn_all, all_d)
    save_fr_d(fn_onewrd, onewrd_d)
    save_fr_d(fn_lowercase, lowercase_d)

    print("Done.")


def build_balda_dic():  # posL=None):
    print("Building Balda dictionary...")
    posL = None
    """
	print("Building Balda dictionary with parts of speech:", posL)
	synsets_in_wordnet = wordnet.all_synsets()
	letters = set('abcdefghijklmnopqrstuvwxyz')
	n = 117658
	frac = 10
	all_d = {} #all words and expressions with definitions
	onewrd_d = {}
	lowercase_d = {} 
	for i, s in enumerate(synsets_in_wordnet):
		if i > n * frac / 100:
			print(f"{frac}% done.")
			frac += 10

		if posL and s.pos() not in posL: 
			continue 

		for wrd in s.lemma_names():
			#all_d[wrd] = 0
			if '_' in wrd: continue
			#onewrd_d[wrd] = 0
			#if len(wrd) == 1: continue
			for c in wrd:
				if c not in letters: break 
			else:
				lowercase_d[wrd] = 0

	print("Building frequency dictionary...")
	"""
    fr_l, fr_d = load_freq_dic(fn_fr_d)

    """
	for wrd, freq in fr_d.items():
		if wrd in lowercase_d:
			lowercase_d[wrd] = freq

	print("Scrabblizing...")"""
    scr_l = load_lines('sowpods.txt')
    res_d = {}
    for w in scr_l:
        """w = w.lower()
        if w in lowercase_d:
                res_d[w] = lowercase_d[w]
        else:
                if not posL:
                        res_d[w] = -1 #scr
        """
        try:
            res_d[w] = fr_d[w]
        except:
            res_d[w] = 0
    print("Saving dictionary...")
    fn_balda = makefn_balda(posL)
    save_fr_d(fn_balda, res_d)


def wordlist2fr_d(fnwordlist, fnnew):
    print("Building freq dictionary...")

    fr_l, fr_d = load_freq_dic(fn_fr_d)

    scr_l = json.loads(read_all(fnwordlist))
    res_d = {}
    for w in scr_l:
        try:
            res_d[w] = fr_d[w]
        except:
            res_d[w] = 0
    print("Saving dictionary...")
    save_fr_d(fnnew, res_d)


def my_lemmatize_fromscratch(wrd, parts_of_speech=None) -> dict[str, set[str]]:
    '''Find lemmas for the wrd, broken down by parts of speech (keys in resulting dict)

    For example, if wrd is 'lay' then res['v'] will be {lie, lay} because lay can either be
    the past tense of lie or the infinitive lay'''
    # synset is basically a specific meaning (several words could have the same meaning and thus
    # belong to the same synset, and of course a word can have different meanings, so could
    # belong to different synsets).
    synset_db = wordnet._lemma_pos_offset_map
    # maps word (lemma) to its meanings/synsets
    # for each word gives dictionary like:
    # {'a': [3478, 754289],...} Where for each part of speech you get a list of synsets (their ids)
    get_synset = wordnet.synset_from_pos_and_offset  # to get from ss id to actual ss

    if parts_of_speech is None:
        parts_of_speech = 'nvasr'
    resD = {}
    for pos in parts_of_speech:
        resD[pos] = set()
        for maybe_lemma in wordnet._morphy(wrd, pos):  # candidate lemmas
            # that's what wordnet.synsets uses, which results in wrd="beses" being considered a
            # form of lemma 'be' so we check that our
            # maybe lemma can possibly be the part of speech we are now considering
            # for example be is not a noun, so the candidate lemma 'be' for 'beses' (with pos=noun)
            # will not pass through the next if statement, so won't be added to the results
            if pos in synset_db[maybe_lemma]:
                # could add the candidate lemma to results, but another problem is that it may
                # have incorrect capitalization (wrd=junes can produce candidate june)
                # so to try to get correct capitalization we look up the synsets it belongs to
                # (I guess synset_db doesn't care about capitalization?) and then look up
                # the words in each ss, keep only the ones that match and use their
                # capitalization. For example, if wrd is 'was', one of the maybe_lemma-s is wa
                # (for better or worse - TODO investigate), then looking up its ss we get the one
                # that contains {Washington, WA} so we see that the correct capitalization is WA
                # TODO but maybe I shouldn't keep words that have a different capitalization
                # from the original - investigate
                if 1:  # without this everything will be lowercase
                    found = False
                    for ssid in synset_db[maybe_lemma][pos]:
                        ss = get_synset(pos, ssid)
                        #
                        for lemma in ss.lemma_names():
                            if lemma.lower() == maybe_lemma:
                                resD[pos].add(lemma)  # for example wa becomes WA
                                # found = True
                                # break
                        # if found: break
                # resD[p].add(form)
    return resD


def build_lemmatization_dic(fn):
    scr_l = load_lines('sowpods.txt')
    res_l = []
    i = 0
    for wrd in scr_l:
        i += 1
        if not i % 1000:
            print(i)
        lemD = my_lemmatize_fromscratch(wrd)
        res_l.append((wrd, lemD))
    with open(fn, 'w') as f:
        for r in res_l:
            lemstr = '; '.join(
                [
                    f"{pos}. {', '.join(lemset)}"
                    for pos, lemset in r[1].items()
                    if lemset
                ]
            )
            f.write(f"{r[0]}: {lemstr}\n")
            # input(f"{r[0]}: {lemstr}\n")


from vocabutils import read8based, read_all, readopen


def scrapewiktionary(fn_save, fn_wordlist=None):
    # x = 'zēlotypia' can't be handled by default encoding used by open
    # (the default is utf8 for converting between str and byte, but for open it uses
    # print(locale.getpreferredencoding())
    # update: i set environmental variable PYTHONUTF8 = 1 and included a check in the beginning

    fn_cur = "word about to be scraped.txt"
    fn_failed = "words that could not be fetched.txt"

    capitals = 0  # input('Do you want me to capitalize words before looking them up? (y/n)') == 'y'
    # if want to start from scratch
    with open(fn_cur, 'w') as fc:
        pass
    with open(fn_save, 'w') as fc:
        pass
    with open(fn_failed, 'w') as fc:
        pass

    from wiktionaryparser import WiktionaryParser

    parser = WiktionaryParser()

    if fn_wordlist:
        # fr_l = load_lines(fn_wordlist)
        with open(
            fn_wordlist, 'r'
        ) as f:  # any encoding is fine, cuz only ascii is used by dflt
            fr_l = json.load(
                f
            )  # but this can now contain unicode chars like 'zēlotypia'
    else:
        fr_l, fr_d = load_freq_dic(fn_balda)
        fr_l = [x[0] for x in fr_l]

    cnt = 0
    skip = True
    wcur = read_all(fn_cur)

    frs = set()
    for wrd in fr_l:
        if wrd in frs:
            continue
        frs.add(wrd)
        if capitals:
            wrd = wrd.capitalize()
        cnt += 1
        print(f"{cnt}. {wrd}... ", end='')

        if skip:
            if not wcur or wrd == wcur:
                skip = False
            else:
                print('Skipping.')
                continue

        with open(fn_cur, 'w') as fc:
            fc.write(wrd)

        try:
            word = parser.fetch(wrd)
        except:
            word = None

        if not word:
            with open(fn_failed, 'a') as ff:
                ff.write(wrd + '\n')
            print('Faled to fetch.')
            continue
        if not word[0]['definitions']:
            print('Fetched but no definitions, not saving.')
            continue
        print(f"Fetched. Saving... ", end='')

        wordst = json.dumps({wrd: word}, indent=4)[1:-2] + ', \n'
        # print(wordst)
        with open(fn_save, 'a') as f:
            f.write(wordst)
        print('Saved.')

    with open(fn_cur, 'w') as fc:
        pass


def process_wikdict(fn, fnnew):
    """converts full wiktionary records into compact definitions"""
    with open(fnnew, 'w') as f:
        pass

    with open(fn, 'r') as f:
        jsn = f.read()
    if jsn[0] != '{':  # true unless I added it manually
        jsn = '{' + jsn.strip().strip(',') + '}'

    wdic = json.loads(jsn)
    newdic = {}
    cnt = 0
    for k, v in wdic.items():
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        sepsenseL = []
        for sepsense_d in v:
            if "definitions" not in sepsense_d:
                continue
            posL = []
            for pos in sepsense_d["definitions"]:
                posS = pos["partOfSpeech"]
                txtL = pos["text"]
                wrdinfo = txtL[0]
                posdefL = txtL[1:]
                # posL.append((posS, wrdinfo, posdefL))
                posL.append({'pos': posS, 'info': wrdinfo, 'defs': posdefL})

            if posL:
                sepsenseL.append(posL)

        if sepsenseL:
            newdic[k] = sepsenseL

    newdicS = json.dumps(newdic)
    # 	newdicS = json.dumps(newdic, indent=2)
    with open(fnnew, 'a') as f:
        f.write(newdicS)
    print(len(newdic), 'compact definitions saved.')


def narrow_wikdefs_to_lowercase_wordnet(fn, fnnew):
    with open(fnnew, 'w') as f:
        pass
    fr_l, fr_d = load_freq_dic(fn_lowercase)

    with open(fn, 'r') as f:
        jsn = f.read()
    wdic = json.loads(jsn)
    newdic = {}
    cnt = 0
    for k, v in wdic.items():
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        if k in fr_d:
            newdic[k] = v

    newdicS = json.dumps(newdic)
    # 	newdicS = json.dumps(newdic, indent=2)
    with open(fnnew, 'a') as f:
        f.write(newdicS)


def remove_nodef(fn_freqdict, fn_defs, fnnew):
    """takes file in frequency dictionary format and a file with definitions and removes words
    from the frequency dictionary that don't have definitions"""
    frl, frd = load_freq_dic(fn_freqdict)
    with open(fn_defs, 'r') as f:
        defD = json.load(f)
    frlnew = [x for x in frl if x[0] in defD and 'definitions' in defD[x[0]][0]]
    save_as_str(fnnew, frlnew)
    print(len(frl), len(frlnew))


def find_unhelpful_defs(fn_defs, fnnew, fn_tolookup):
    with open(fn_defs, 'r') as f:
        defD = json.load(f)
    betterformsD = {}
    betterformsL = []
    for w, defL in defD.items():
        if len(defL) > 1:
            continue
        posL = defL[0]
        if len(posL) > 1:
            continue

        dfns = posL[0]['defs']
        if len(dfns) > 1:
            continue
        dfn = dfns[0]
        # Of or pertaining to
        # In a...Manner,  Like a , One who, That which , capable of being , degree of being, able to be,
        # The ability to be
        soiL = [
            'lural of ',
            've form of ',
            'participle of ',
            'spelling of ',
            'singular of ',
            'Synonym of ',
            'bsolete form of ',
            'istoric form of ',
        ]
        otherL = [
            'elating to ',
            ' tate of being ',
            'uality of being ',
            'uality of not being ',
        ]
        terminators = list(')(,;:[.') + ['See ', ' -']

        newwrd = dfn
        for soi in soiL:
            if soi in newwrd:
                newwrd = st_until(newwrd.split(soi)[1], terminators).strip()

        if newwrd != dfn and newwrd.count(' ') <= 1:
            # print(w, newwrd, ', def:', dfn)
            betterformsD[w] = newwrd
            if newwrd not in defD:
                betterformsL.append(newwrd)

    with open(fnnew, 'w') as f:
        json.dump(betterformsD, f)

    with open(fn_tolookup, 'w') as f:
        json.dump(betterformsL, f)

    print(
        'Done.',
        len(defD),
        'total definitions examined.',
        len(betterformsD),
        'better spellings found.',
        len(betterformsL),
        'words to look up.',
    )


def combine_jsondicts(fnMain, fnToAdd):
    wd = json.loads(read_all(fnMain))
    wdb = json.loads(read_all(fnToAdd))
    print(len(wd), len(wdb))
    for k, v in wdb.items():
        wd[k] = v
    with open(fnMain, 'w') as f:
        json.dump(wd, f)
    print(len(wd))


class def2refdata:
    # Of or pertaining to
    # In a...Manner,  Like a , One who, That which , capable of being , degree of being, able to be,
    # The ability to be
    formOfL = [
        'Plural of ',
        'Indicative form of ',
        'Alternative form of ',
        'Comparative form of ',
        'Superlative form of ',
        'participle of ',
        'spelling of ',
        'singular of ',
        'Obsolete form of ',
        'Historic form of ',
    ]
    otherL = [
        'Synonym of ',
        'Relating to ',
        'State of being ',
        'Quality of being ',
        'Quality of not being ',
    ]
    moreL = [
        '#Of or pertaining to ',
        '#Pertaining to ',
        '#Like ',
        '#One who ',
        '#That which ',
        '#Capable of being ',
        '#Degree of being ',
        '#Able to be ',
        '#Ability to be ',
        '#The ability to be ',
    ]
    # TODO: for "one who", "that which" there will be an extra 's' at the end
    # in a... Manner
    # don't just check that the referenced word is in defD, check it's the right pos
    soiL = formOfL + otherL + moreL
    reftypeL = ['pl', 'tm', 'af', 'co', 'su', 'tm', 'sp', 'si', 'af', 'af'] + [
        'sy',
        're',
        'be',
        'be',
        'nb',
        'of',
        'of',
        'li',
        '1w',
        'tw',
        'ab',
        'de',
        'ab',
        'bb',
        'bb',
    ]
    altform_refs = ['pl', 'tm', 'af', 'co', 'su', 'tm', 'sp', 'si']
    altform_refs += [x.upper() for x in altform_refs]

    terminators = list(')(,;:[.') + ['See ', ' -']
    endingpairs = [
        ('city', 'ciousness'),
        ('ity', 'ness'),
        ('ty', 'ness'),
        ('ous', 'ic'),
        ('ical', 'ic'),
        ('al', 'ic'),
        ('istic', 'ic'),
        ('ar', 'tic'),
        ('ist', 'er'),
        ('ism', 'age'),
        ('ist', 'ian'),
        ('al', 'ic'),
        ('ar', 'atory'),
        ('ism', 'ianism'),
        ('ly', 'like'),
        ('y', 'ie'),
        ('ism', 'y'),
        ('gue', 'gist'),
        ('ory', 'ive'),
        ('ish', 'y'),
        ('ing', 'al'),
        ('y', 'esis'),
        ('ary', 'al'),
        ('ary', 'ist'),
        ('ary', 'ine'),
        ('y', 'ing'),
        ('ate', 'ar'),
        ('ated', 'ar'),
        ('try', 'cy'),
        ('ing', 'ion'),
        ('ing', 'ent'),
        ('ing', ''),
        ('ist', ''),
        ('tus', 't'),
        ('tsa', 'na'),
        ('ose', 'al'),
        ('ose', 'ar'),
        ('ing', 'ation'),
        ('ingly', 'ingly'),
        ('ly', 'ingly'),
        ('um', ''),
    ]
    endingpairs += [(ee, e) for e, ee in endingpairs]


"""from krovetzstemmer import Stemmer 
kstemmer = Stemmer()"""


def def2ref(wrd, dfn, defD):
    d = def2refdata()
    # if wrd[0] > 's': return'',''
    orig = dfn
    infoL = []
    dfn = dfn.strip()
    if dfn.startswith('(') and ')' in dfn:
        dfn, infoL = process_init_brackets(dfn)

    newwrd = dfn
    ref = ''
    for soi, reftype in zip(d.soiL, d.reftypeL):
        if soi[0] == '#':
            soi = soi[1:]
            atbeg = True
        else:
            atbeg = False

        if soi in newwrd or soi.lower() in newwrd:
            if soi not in newwrd:
                soi = soi.lower()
            if atbeg and newwrd.find(soi) > 1:
                continue
            newwrd = st_until(newwrd.split(soi)[1], d.terminators).strip()
            if ref == '':
                ref = reftype
            """
			if soi== 've form of ':
				print('---',wrd, 'newwrd:', newwrd, 'orig:',dfn)
				input()"""
    newwrd = st_cutbeg(
        newwrd, ['A ', 'a ', 'The ', 'the ', 'An ', 'an ', 'To ', 'to ']
    ).strip()

    if newwrd != dfn and newwrd in defD:
        ref = ref.upper()
        # print(wrd, 'newwrd:', newwrd, 'orig:',dfn); input()

    if ref:
        return ref, newwrd
    # check if the definition is pretty much just the same word, like fantastic and fantastical
    if not ref:
        dfn = st_cutbeg(
            dfn, ['A ', 'a ', 'The ', 'the ', 'An ', 'an ', 'To ', 'to ']
        ).strip()
        dfn = dfn.strip(' \n\t\r.,;')
        if dfn and dfn[0].isupper():  # and dfn not in defD:
            dfnl = dfn.lower()
            if dfnl in defD:
                # TODO: permanently decap
                # print('decapitalyzing', dfn, dfnl);# input()
                dfn = dfnl

        dfnInD = dfn in defD
        if not dfnInD:
            return '', ''

        if {'British spelling'} & set(infoL):
            return 'SP', dfn
        if {'obsolete', 'archaic', 'dated'} & set(infoL):
            return 'OL', dfn
        if {'dialect'} & set(infoL):
            return 'DI', dfn
        if {'informal', 'colloquial'} & set(infoL):
            return 'IN', dfn
        if {'slang'} & set(infoL):
            return 'SL', dfn
        if {'nonstandard'} & set(infoL):
            return 'NS', dfn

        # ws = set(wrd)
        # ds = set(dfn)
        # notincommon = ws ^ ds
        """try:
			dfnstm = kstemmer.stem(dfn)
			wrdstm = kstemmer.stem(wrd)
		except:
			dfnstm = dfn[:-1]
			wrdstm = wrd[:-1]"""
        wrd2, dfn2 = st_cutends(wrd, dfn, d.endingpairs)
        maxlen = max(len(dfn), len(wrd))
        mm = 0
        m = 0
        eddist = nltk.edit_distance(dfn2, wrd2)
        if eddist < min(5, 0.4 * maxlen):
            mm = 1
        if eddist < min(5, 0.25 * maxlen):
            m = 1
        if (
            eddist < 4
            and ' ' not in dfn
            and min(len(wrd), len(dfn)) >= 6
            and wrd[:6] == dfn[:6]
        ):
            m = 1
        # if dfn and len(notincommon) < 4:#ordered set?
        ref = 'SI' if m else 'SY'
        if 0:
            print(
                wrd,
                'dfn:',
                dfn,
                'orig:',
                orig,
                m,
                mm,
                'eddist:',
                dfn2,
                wrd2,
                nltk.edit_distance(dfn2, wrd2),
                'ref:',
                ref,
            )
            input()
        return ref, dfn


def build_wikdefs_withrefs(fn_defs, fnnew):
    """take file with compact definitions and add references from less standard forms
    to more standard, so the structure now is:
    keys pos, info, defs and now refs: [(reftype, morestandardword), (another reftype,..),...],
    which has the same number of elements as defs (if a particular def is not a ref the corresponding
    entry should be ('',''))
    """
    with open(fn_defs, 'r') as f:
        defD = json.load(f)
    tolookupL = []
    betterformsL = []
    cnt = 0
    for w, sepsenseL in defD.items():
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        nodefs = True
        norefs = True
        for posL in sepsenseL:
            for pos in posL:
                if 'defs' not in pos:
                    continue
                dfns = pos['defs']
                refL = []
                for dfn in dfns:
                    nodefs = False
                    ref, newwrd = def2ref(w, dfn, defD)
                    refL.append([ref, newwrd])
                    if not ref:
                        continue

                    norefs = False
                    if newwrd not in defD:
                        tolookupL.append(newwrd)  # to look up later
                        # print(newwrd)#; input()
                    betterformsL.append([w, ref, newwrd])

                # add 'refs' field
                if not norefs:
                    pos['refs'] = refL
                    import random

                    if (
                        pos['refs']
                        and pos['refs'][0][0] != 'PL'
                        and random.random() < -0.001
                    ):  # and wrd
                        print(pos)
                        input()

    with open(fnnew, 'w') as f:
        json.dump(defD, f)

    fn_tolookup = 'to look up.txt'
    with open(fn_tolookup, 'w') as f:
        json.dump(tolookupL, f)
        # json.dump(betterformsL, f)

    print(
        'Done.',
        len(defD),
        'total definitions examined.',
        len(betterformsL),
        'references found.',
        len(tolookupL),
        'words to look up.',
    )


def st_until(st, terminators):
    res = st
    for t in terminators:
        if t in res:
            res = res.split(t)[0]
    return res


def st_cutbeg(dfn, begs):
    for beg in begs:
        if dfn.startswith(beg):
            return dfn[len(beg) :]
    return dfn


def st_cutends(wrd, dfn, endspairsL):
    wrdlonger = len(wrd) - len(dfn)
    for e, ee in endspairsL:
        elonger = len(e) - len(ee)
        if wrdlonger * elonger < 0:
            continue
        if wrd.endswith(e) and dfn.endswith(ee):
            if e:
                wrd = wrd[: -len(e)]
            if ee:
                dfn = dfn[: -len(ee)]
    return wrd, dfn


def process_init_brackets(w):
    wspl = w.split(')')
    wrd = wspl[1].strip()
    infoL = [x.strip() for x in wspl[0][1:].split(',')]
    return wrd, infoL


def build_good_wikdefs(fn_withrefs, fnnew, fnwordlist):
    d = def2refdata()
    defD = json.loads(read_all(fn_withrefs))
    cnt = 0
    newD = {}
    wL = []
    for w, sepsenseL in defD.items():
        cnt += 1
        good = False
        if cnt % 1000 == 0:
            print(cnt)
        for posL in sepsenseL:
            for pos in posL:
                if 'defs' not in pos:
                    continue
                if pos['pos'].lower() == 'proper noun':
                    continue
                if 'refs' not in pos:
                    good = True
                    continue
                dfns = pos['defs']
                for dfn, ref in zip(dfns, pos['refs']):
                    if ref[0] not in d.altform_refs:
                        good = True
                        break
                # need to later include a mechanism to pull up referenced words and show them too
                if good:
                    break
            if good:
                break
        if good:
            newD[w] = sepsenseL
            wL.append(w)

    with open(fnnew, 'w') as f:
        json.dump(newD, f)
    with open(fnwordlist, 'w') as f:
        json.dump(wL, f)
    print(len(wL))


from vocabutils import check_default_file_encoding

check_default_file_encoding()
fn_fr_d = 'frequency dictionary from NLTK corpora.txt'  # 27068
fn_all = 'all words and expressions with Wordnet definitions.txt'
fn_onewrd = 'all words with Wordnet definitions.txt'
fn_lowercase = 'all lowercase words with Wordnet definitions.txt'
fn_balda = makefn_balda(None)
fn_lemmatize = 'lemmatizations.txt'
# the difference between wiktionary.txt and wiktionary definitions.txt is that
# the former has the full wiktionary entries, but the latter has only compact definitions,
# obtained from the former by process_wikdict().
fn_wik = "wiktionary.txt"
fn_wikdefs = "wiktionary definitions.txt"
fn_wikdefslower = "wiktionary definitions for lowercase wordnet.txt"
fn_lowercase_withwikdefs = (
    "all lowercase words with Wordnet definitions and Wik definitions.txt"
)
fn_betterforms = "better spellings of words.txt"
fn_tolookup = "better spellings list to look up.txt"
fn_wik_betterspellings = 'wik better spellings.txt'
fn_wikdefs_betterspellings = 'wik defs better spellings.txt'
fn_wikdefs_withrefs = "wiktionary definitions with references.txt"
fn_wordlistgood = 'good words.txt'
fn_wikdefsgood = 'wik defs good.txt'
fn_freqdictgood = 'freq dict good.txt'

# The procedures are in reverse order, in case I need to rebuild stuff

wordlist2fr_d(fn_wordlistgood, fn_freqdictgood)
# build_good_wikdefs(fn_wikdefs_withrefs, fn_wikdefsgood, fn_wordlistgood)
# build_wikdefs_withrefs(fn_wikdefs, fn_wikdefs_withrefs); exit()
# find_unhelpful_defs(fn_wikdefs, fn_betterforms, fn_tolookup)
# remove_nodef(fn_lowercase, fn_wik, fn_lowercase_withwikdefs);
# process_wikdict(fn_wik, fn_wikdefs); exit()
# process_wikdict(fn_wik_betterspellings, fn_wikdefs_betterspellings); exit()
# narrow_wikdefs_to_lowercase_wordnet(fn_wikdefs, fn_wikdefslower); exit()
# scrapewiktionary(fn_wik_betterspellings, fn_tolookup); exit()
exit()

words = set(nltk.corpus.words.words())
lemmer = nltk.stem.WordNetLemmatizer()

# buildFreqDic(fn_fr_d)
# build_dic(); exit()
# build_balda_dic(); exit()
# build_lemmatization_dic(fn_lemmatize); exit()
