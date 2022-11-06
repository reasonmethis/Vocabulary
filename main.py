import random
import time

from vocabutils import load_freq_dic, load_lines, makefn_balda, wait_for_option
from wordinfo import formatted_meanings

baldaminwrdlen = 3
verbose = False  # if true, display info about buckets, stdevs, etc.

# words_l = nltk.corpus.words.words()
# words = set(words_l)
# fn_fr_d = 'frequency dictionary from NLTK corpora.txt' #27068
fn_all = 'all words and expressions with Wordnet definitions.txt'
fn_onewrd = 'all words with Wordnet definitions.txt'
fn_lowercase = 'all lowercase words with Wordnet definitions.txt'
fn_balda = makefn_balda(None)
fn_freqdictgood = 'freq dict good.txt'
# fn_lemmatize = 'lemmatizations.txt'
# fn_wik = "wiktionary.txt"

class Baldastate:
    def __init__(self, wrd, fpturn, words, bestseq=None):
        self.wrd = wrd
        self.firstplayersturn = fpturn
        self.possiblewords = words
        self.bestseq = bestseq

    def is_word(self):
        return self.wrd in self.possiblewords

    def leafscore(self):
        tmp = 1 - 0.001 * len(self.wrd)
        return tmp if self.firstplayersturn else -tmp


def build_possiblewords(wrd, words):
    words = sorted(words)
    res = []
    wcur = '=%$¥'
    for word in words:
        # if len(w) == 1: print(w)
        # if fr_d[w] < 1: continue
        if not word.startswith(wcur) and (word.startswith(wrd)):
            wcur = word
            res.append(word)
    random.shuffle(res)
    # print(len(res))
    return res


def get_score(cur_state: Baldastate, goal_stop):
    verbose = 0
    if verbose:
        print(
            '-' * len(cur_state.wrd) * 2 + 'examining',
            cur_state.wrd,
            len(cur_state.possiblewords),
            end=' ',
        )
    nxt_states, best_score = get_next(cur_state)
    if verbose:
        print(
            'getnext: ',
            best_score,
            len(nxt_states) if nxt_states is not None else nxt_states,
        )

    if best_score is not None:
        cur_state.bestseq = cur_state.wrd
        return best_score, None

    best_score = -1 if cur_state.firstplayersturn else 1
    if not nxt_states:  # i don't know what word it could be, i lose
        return best_score, None

    best_new_state = None
    scorecumul = 0
    nbranches = 0
    for state in nxt_states:
        # if len(cur_state.wrd)==0 and state.wrd!='o': continue
        score, _ = get_score(state, best_score)
        scorecumul += score
        nbranches += 1
        # if len(state.wrd) < 4:
        # print(f"{state.wrd}({score})", end="  ")
        # time.sleep(0.4)
        modifactor = 0.1
        if 1 or (score < 0) == cur_state.firstplayersturn:
            # if comp is heading for a word it already lost on, keep score at minimum
            for lw in binfo.losingwords:
                if lw.startswith(state.wrd):
                    modifactor = 0.01
                    break
        # if state.wrd == "ju": print(dontchange, binfo.losingwords, score)
        score = score + modifactor * (scorecumul / nbranches - score)

        best_score, code = compare_scores(score, best_score, goal_stop, cur_state)
        if code or (best_new_state is None):
            best_new_state = state
        if code == 2:
            break
    if verbose:
        print(
            '-' * len(cur_state.wrd) * 2 + 'done with',
            cur_state.wrd,
            'score:',
            best_score,
            goal_stop,
        )
    cur_state.bestseq = best_new_state.bestseq
    return best_score, best_new_state


def compare_scores(score, best_score, goal_stop, state: Baldastate):
    if state.firstplayersturn:
        if score > best_score:
            # if flag == 1: input('oops')
            return score, 2 if score >= goal_stop else 1
    else:
        if score < best_score:
            # if flag == 1: input('oops')
            return score, 2 if score <= goal_stop else 1
    return best_score, 0


def get_next(state: Baldastate):
    '''Returns a pair: list of next possible Baldastate-s (one per possible next letter, each
    storing a full list of possible words compatible with it) AND leaf score if the state is terminal
    (word completed)'''
    if state.is_word():
        return None, state.leafscore()
    sz = len(state.wrd)
    letters: dict[str, Baldastate]= {}  # new possible states
    for wrd in state.possiblewords:
        if (lt := wrd[sz]) in letters:
            letters[lt].possiblewords.append(wrd)
        else:
            letters[lt] = Baldastate(state.wrd + lt, not state.firstplayersturn, [wrd])
    return letters.values(), None


class Balda_session_info:
    def __init__(self):
        self.addedwords = []
        self.losingwords = []
        self.ncompwins = 0
        self.nplrwins = 0


def play_balda(allwords, compfirst=False, initwrd=""):
    msgssz = 42
    plrturn = compfirst
    compwinsscore = 1 if compfirst else -1
    wrd = initwrd
    allwords += binfo.addedwords
    while True:
        plrturn = not plrturn
        if plrturn:
            if wrd:
                c = input(
                    'Add one letter (or "quit" to quit): '.ljust(msgssz) + wrd + '... '
                )
            else:
                c = input('Enter the first letter (or "quit" to quit): '.ljust(msgssz))

            if c in ['exit', 'quit', 'bye']:
                return 'quit'

            if not c:
                print()
                if wrd in fr_d and len(wrd) >= baldaminwrdlen:
                    print(wrd, 'is a word, you win!\n')
                    if wrd not in allwords:
                        binfo.addedwords.append(wrd)
                        # allwords.append(wrd)
                    else:
                        binfo.losingwords.append(wrd)
                    return 'plrwins'
                if wrd in fr_d:
                    print(
                        f"{wrd} is a word but only words at least {baldaminwrdlen} letters long count."
                    )
                else:
                    print(wrd, 'is not a word in the Scrabble dictionary.')
                """tmp = input('Type anything to continue or press ENTER to quit: ')
				print()
				if not tmp:
					return 'quit'
				plrturn = False #to keep player's turn
				continue
			if c == '?':"""
                posswords = build_possiblewords(wrd, allwords)
                print(
                    'Possible words: ' + ', '.join(posswords[: min(10, len(posswords))])
                )
                print('I win!\n')
                return 'compwins'
            if c.startswith('?'):
                if c == '?':
                    print(
                        'To finish the round press ENTER instead of entering a letter. To find out the meaning of a word, enter it instead of one letter. Type quit or exit to quit the game.'
                    )
                else:
                    qry = c[1:]
                    print(qry in allwords, fr_d[qry] if qry in fr_d else None)
                plrturn = False  # to keep player's turn
                print()
                continue
            if len(c) > 1:
                mns = formatted_meanings(c, strictmatch=False)
                if mns:
                    print("Meanings of", c + ':')
                    print(mns)
                elif c in fr_d:
                    print(
                        f"I don't know what {c} means but I know it's in the Scrabble dictionary."
                    )
                else:
                    print(c, 'is not a word in the Scrabble dictionary.')

                plrturn = False  # to keep player's turn
                print()
                continue
            wrd += c
            print()
            continue
        for j in range(1):  # for testing
            posswords = build_possiblewords(wrd, allwords)
            state = Baldastate(wrd, compfirst, posswords)
            score, nxt = get_score(state, 1 if compfirst else -1)
            # print("First player's expected score:", round(score, 2), "Best sequence:", state.bestseq)

        if not nxt:
            if score * compwinsscore > 0:  # same sign
                if wrd not in fr_d:
                    print('Error.')
                    score, nxt = get_score(state, 1)
                else:
                    print(wrd, 'is a word I know, I win!\n')
                mns = formatted_meanings(wrd, strictmatch=False)
                if mns:
                    print("Meanings of", wrd + ':')
                    print(mns)
                else:
                    print(
                        "I don't know what it means but I know it's in the Scrabble dictionary."
                    )
                print()
                return 'compwins'
            else:
                while True:
                    w = input(
                        "I give up. To win, tell me a word that begins with "
                        + wrd
                        + " or press ENTER to resign: "
                    )
                    if not w:
                        print('I win!\n')
                        return 'compwins'
                    if not w.startswith(wrd):
                        print(f"That doesn't start with {wrd}.")
                        continue
                    if len(w) < baldaminwrdlen:
                        print(
                            f"This word is too short, words must be at least {baldaminwrdlen} letters long."
                        )
                        continue
                    if w in fr_d:
                        print('You win!\n')
                        binfo.addedwords.append(w)
                        # allwords.append(w)
                        return 'plrwins'
                    print('There is no such word in the Scrabble dictionary.')
            print()
            return ''
        wrd = (
            nxt.wrd if len(nxt.wrd) > 1 else random.choice('abcdefghijklmnopqrstuvwxyz')
        )
        if len(wrd) == 1:
            print("I'll go with the letter:".ljust(msgssz), end='')
        else:
            print("I'll add one letter:".ljust(msgssz) + wrd[:-1] + '... ', end='')
        time.sleep(0.5)
        print(wrd[-1])


def get_difs(int_l):
    """get a list of differences between elements of a given list"""
    res_l = []
    n_prev = None
    for n in int_l:
        if n_prev is not None:
            res_l.append(n - n_prev)
        n_prev = n
    return res_l


def load_lemmatize_d(fn):
    lines = load_lines(fn)
    resD = {}
    for st in lines:
        st2 = st.split(': ')
        st3 = st2[1].split('; ')
        posD = {}
        for posentry in st3:
            tmp = posentry.split('. ')
            lemmas = set(tmp[1].split(', ')) if posentry else set()
            posD[tmp[0]] = lemmas
        resD[st2[0]] = posD
    return resD


def group_beg(wrdi):  # first index with the same frequency
    fr = fr_l[wrdi][1]
    i = wrdi - 1
    while i >= 0 and fr_l[i][1] == fr:
        i -= 1
    return i + 1


def group_end(wrdi):
    fr = fr_l[wrdi][1]
    i = wrdi + 1
    while i < n_tot and fr_l[i][1] == fr:
        i += 1
    return i - 1


def get_group(wrdi, minsize=1):
    i1 = group_beg(wrdi)
    i2 = group_end(wrdi)
    if (siz := i2 - i1 + 1) >= minsize:
        return i1, i2
    dsize = minsize - siz
    i1 = max(0, i1 - dsize // 2)
    i2 = min(n_tot - 1, i1 + minsize - 1)
    return i1, i2


def gen_buks(fr_l):
    # n_non0 = 27068 #number of words with frequency non-zero
    fr_uneq = 5
    n_b = 15  # _b here and elsewhere stands for bucket

    # find the first word with frequency fr_uneq and remember its index
    for i_i, x in enumerate(fr_l):
        if x[1] <= fr_uneq:
            break

    buk_l = [None] * n_b
    fr_prev = -1
    cur_b = n_b - fr_uneq - 1  # this buk will be for words with frequency fr_uneq
    n_b1 = cur_b  # previous buckets will split higher frequency words equally

    # for each bucket, we store the index of the first word that belongs to the bucket
    # for buckets starting from cur_b, we put words in the same bucket if they have the same freq
    for i in range(i_i, n_tot):
        if fr_l[i][1] != fr_prev:
            buk_l[cur_b] = i
            fr_prev = fr_l[i][1]
            cur_b += 1
    # in case the assumption that each next freq is one less the previous,
    # with the last one being zero, is wrong, fill in remaining buckets with 0 words
    while cur_b < n_b:
        buk_l[cur_b] = n_tot
        cur_b += 1
    # i1 = buk_l[n_b1]
    # fill in earlier buckets, for higher frequency words
    for i in range(n_b1):
        buk_l[i] = int(i * i_i / n_b1 + 0.5)
    return buk_l


def test_vocab():
    tries_per_buk = [0] * n_b
    score_per_buk = [0] * n_b

    print("-" * 100)
    while True:
        dvar_l = []
        tries_per_buk1, score_per_buk1 = modify_heuristicly(
            tries_per_buk, score_per_buk
        )
        vocab_estimate, var_cumul = calc_vocab(tries_per_buk1, score_per_buk1)
        # vocab_estimate, var_cumul = calc_vocab(tries_per_buk, score_per_buk)
        if user_voc_est != -1:
            vocab_estimate, var_cumul = user_voc_est, user_stdev**2

        for i in range(n_b):
            n_eff = tries_per_buk1[i] + 1
            prob = (score_per_buk1[i] + 0.5) / n_eff

            tries_per_buk[i] += 1
            tries_per_buk2, score_per_buk2 = modify_heuristicly(
                tries_per_buk, score_per_buk, i, 0
            )
            _, var_cumul0 = calc_vocab(tries_per_buk2, score_per_buk2)

            score_per_buk[i] += 1
            tries_per_buk2, score_per_buk2 = modify_heuristicly(
                tries_per_buk, score_per_buk, i, 1
            )
            _, var_cumul1 = calc_vocab(tries_per_buk2, score_per_buk2)

            tries_per_buk[i] -= 1
            score_per_buk[i] -= 1

            var_cumul_new_expected = prob * var_cumul1 + (1 - prob) * var_cumul0
            if buk_n_l[i]:
                dvar_l.append(var_cumul - var_cumul_new_expected)
            else:
                dvar_l.append(0)

        print(
            f"\nYOUR VOCABULARY: {int(vocab_estimate + 0.5)} ± {int(var_cumul ** 0.5 + 0.5)}."
        )
        if verbose:
            print(score_per_buk)
            print(tries_per_buk)
            print(score_per_buk1)
            print(tries_per_buk1)
            print("Expected new stdevs:")
            tmp = [int((var_cumul - x) ** 0.5 + 0.5) for x in dvar_l]
            print(tmp, min(tmp))

        # select bucket and word to show
        if playmode == "improve":
            # wrdi = vocab_estimate + (random.random() - 0.5) * 2 * 0.5 * var_cumul ** 0.5
            # wrdi = user_voc_est + (random.random() - 0.5) * 2 * user_voc_est * 0.2
            wrdi = random.randint(min_i_test, max_i_test)
            minsize = max_i_test - min_i_test + 1
            i, ii = get_group(wrdi, minsize=minsize)
            wrd_i = random.randint(i, ii)
            # find which bucket
            for buk in range(n_b):
                if buk_i_l[buk] > wrdi:
                    buk -= 1
                    break
            # print(wrdi, buk, buk_i_l[buk])
        else:
            buk = dvar_l.index(max(dvar_l))
            wrd_i = buk_i_l[buk] + random.randint(
                0, buk_n_l[buk] - 1
            )  # randint values are inclusive
        wrd = fr_l[wrd_i][0]

        # print(f"DO YOU KNOW: {wrd}".ljust(40) + f"Difficulty: {wrd_i // 100}".ljust(20) + "Your answer (y/n): ", end='')
        print(
            f"DO YOU KNOW: {wrd}".ljust(40)
            + f"Category: {wrd_i}".ljust(27)
            + "Your answer (y/n): ",
            end='',
        )
        ans = wait_for_option({'y': 'yes', 'n': 'no', 'esc': 'quit', 'q': 'quit'})
        print(ans)

        if ans.startswith("q"):
            print("\nThanks for playing!")
            return

        tries_per_buk[buk] += 1
        y = ans.startswith("y")
        if y:
            score_per_buk[buk] += 1
            print(f" ")
        else:
            print(f" ")
        print(formatted_meanings(wrd))
        print("-" * 100)


def modify_heuristicly(tries_per_buk, score_per_buk, buk=None, y=None):
    # modify the scores using heuristic
    tries_per_buk2 = [0] * len(tries_per_buk)
    score_per_buk2 = [0] * len(score_per_buk)
    addper1bukDiff = 20 / (n_b - 1)
    for i in range(n_b):
        nyes = score_per_buk[i]
        nno = tries_per_buk[i] - nyes
        for b in range(i):
            toAdd = addper1bukDiff * (i - b) * nyes
            tries_per_buk2[b] += toAdd
            score_per_buk2[b] += toAdd
        for b in range(i + 1, n_b):
            toAdd = addper1bukDiff * (b - i) * nno
            tries_per_buk2[b] += toAdd

    for i in range(n_b):
        tries_per_buk2[i] += tries_per_buk[i]
        score_per_buk2[i] += score_per_buk[i]
    return tries_per_buk2, score_per_buk2


def calc_vocab(tries_per_buk, score_per_buk):
    var_cumul = 0
    vocab_estimate = 0
    for i in range(n_b):
        n_eff = tries_per_buk[i] + 1
        prob = (score_per_buk[i] + 0.5) / n_eff
        var_contrib = prob * (1 - prob) / n_eff * buk_n_l[i] ** 2
        var_cumul += var_contrib
        vocab_estimate += prob * buk_n_l[i]

    return vocab_estimate, var_cumul

if __name__ =='__main__':
    # buildFreqDic(fn_fr_d)
    # build_dic(); exit()
    # build_balda_dic(); exit()
    # build_lemmatization_dic(fn_lemmatize); exit()

    print('=======================' * 5)
    print()
    print('Welcome to the Vocabulizer 3000!')
    # lemmatize_d = load_lemmatize_d(fn_lemmatize)
    print("Initializing word definitions...\n")
    _ = formatted_meanings("hello")

    while True:
        # Select playing mode
        print("Select mode:")
        print("1 - Measure your vocabulary")
        print("2 - Improve your vocabulary")
        print("3 - Play Balda")
        print("4 - Look up words")
        print('q - Quit')
        print()
        # playmode = 'balda'
        playmode = wait_for_option(
            {
                '1': 'measure',
                '2': 'improve',
                '3': 'balda',
                '4': 'lookup',
                'q': 'quit',
                'esc': 'quit',
            }
        )

        if playmode == 'quit':
            exit()

        if playmode == 'lookup':
            while True:
                w = input('Enter word (q - quit): ')
                if w == 'q':
                    break
                mns = formatted_meanings(w, False)
                print(mns if mns else f"I don't know what '{w}' means.")
                # print(mylemmatize(w))
                print()
            continue

        if playmode == 'balda':
            fr_l, fr_d = load_freq_dic(fn_balda)
            print(
                'I am using a Scrabble dictionary for validation. It has',
                tmp := len(fr_l),
                'words.',
            )
            nwithnonzerofr = 0
            for x in fr_l:
                if x[1] < 1:
                    break
                nwithnonzerofr += 1

            neasy = 5000
            nhard = 100000
            nmedium = 30000  # min(nwithnonzerofr, 30000)
            print('Select the level:')
            print(f'1 - Easy ({neasy} words)')
            print(f'2 - Medium ({nmedium} words)')
            print(f'3 - Hard ({nhard} words)')
            print(f'4 - Impossible (all {tmp} Scrabble words)')
            print('5 - Custom (choose the number of words)')
            print()
            allwords = [x[0] for x in fr_l if len(x[0]) >= baldaminwrdlen and x[1] > 0]
            allwords0 = [x[0] for x in fr_l if len(x[0]) >= baldaminwrdlen and x[1] <= 0]
            random.shuffle(allwords0)
            allwords += allwords0
            nwbalda = wait_for_option(
                {'1': neasy, '2': nmedium, '3': nhard, '4': tmp, '5': 0}
            )

            if not nwbalda:
                while True:
                    nwbalda = int(input(f'Enter how many words to limit myself to: '))
                    if nwbalda > tmp:
                        nwbalda = tmp
                    if nwbalda > 99:
                        break
                    print("C'mon, I should have at least 100 words to work with.")

            print("Using", nwbalda, 'simplest words in my dictionary.\n')
            allwords = allwords[:nwbalda]

            """wordbasicforms = set([x[0] for x in fr_l[:nwbalda]])
            allwords = []
            for x in fr_l:
                for bf in mylemmatize(x[0]):
                    if bf in wordbasicforms:
                        allwords.append(x[0])
                        break"""

            binfo = Balda_session_info()
            compfirst = random.randint(0, 1)
            while True:
                print('-' * 80 + '\n')
                print(
                    f"Let's play Balda! Try to make me finish a word (only words with at least {baldaminwrdlen} letters count).\n"
                )
                print("Current score: You -", binfo.nplrwins, 'Me -', binfo.ncompwins, '\n')
                code = play_balda(allwords, compfirst, "")
                if code == 'quit':
                    print('\n' + '-' * 80 + '\n')
                    break
                if code == 'compwins':
                    binfo.ncompwins += 1
                if code == 'plrwins':
                    binfo.nplrwins += 1
                compfirst = 1 - compfirst
            continue

        '''print("\nSelect vocabulary list to use (recommended - 1):")
        print("1 - Normal words with definitions")
        print("2 - Normal words and proper names with definitions")
        print("3 - All words and expressions with definitions")
        wait_for_option({'1': fn_lowercase, '2': fn_onewrd, '3': fn_all, '4': fn_freqdictgood})'''
        fn_dic = fn_freqdictgood
        fr_l, fr_d = load_freq_dic(fn_dic)

        n_tot = len(fr_l)
        print('\nWords in current list:', n_tot)

        user_voc_est = -1
        if playmode == 'improve':
            while user_voc_est < 0 or user_voc_est > n_tot:
                tmp = input(
                    "To help choose the difficulty of words, tell me how many words you know: "
                )
                user_voc_est = int(tmp)
                user_stdev = user_voc_est * 0.2
                min_i_test = round(max(0, user_voc_est - user_stdev))
                max_i_test = round(min(len(fr_l) - 1, user_voc_est + user_stdev))

        buk_i_l = gen_buks(fr_l)
        buk_n_l = get_difs(buk_i_l)
        buk_n_l.append(len(fr_l) - buk_i_l[-1])
        n_b = len(buk_n_l)
        print(buk_i_l)
        print(buk_n_l)

        test_vocab()

