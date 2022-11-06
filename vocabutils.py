import msvcrt

fn_balda_base = 'frequency dictionary for balda'


def makefn_balda(pos_l):
    if pos_l:
        return fn_balda_base + ' pos ' ' '.join(pos_l) + '.txt'
    return fn_balda_base + '.txt'


def readopen(fn, encoding="utf8"):
    return open(fn, 'r', encoding=encoding) if encoding else open(fn, 'r')


def read_all(fn, encoding='utf8'):
    with readopen(fn, encoding) as f:
        return f.read()


def read8based(fn):
    return readopen(fn, 'utf8' if 'utf8.' in fn else '')


def convert_to_utf8(fn):  # for now just checks
    with open(fn, 'r') as f:
        txt = f.read()
    with open(fn, 'r', encoding='utf8') as f:
        txt8 = f.read()
    print(fn, txt == txt8)


def check_default_file_encoding():
    s = 'zēlotypia'
    import sys

    with open(fn := 'test that utf8 is default for python.txt', 'w') as f:
        f.write(s)
    if s != read_all(fn) or sys.getfilesystemencoding() != 'utf-8':
        raise UnicodeEncodeError


def load_freq_dic(fn):
    # weird entry ('eye-deceiving', 0), ("trompe-l'oeil", 0)
    with open(fn, 'r') as f:
        fr_l = f.read().split('), (')
    fr_l[0] = fr_l[0][2:]
    fr_l[-1] = fr_l[-1][:-2]
    for i, x in enumerate(fr_l):
        x = x.split(", ")
        fr_l[i] = (x[0][1:-1], int(x[1]))
    # print(len(fr_l))
    # print(fr_l[10:])
    fr_d = dict(fr_l)
    return fr_l, fr_d


def load_lines(fn):
    with open(fn, 'r') as f:
        return [line.rstrip('\n\r') for line in f.readlines()]


def wait_for_option(key_d, keypress_detection_mode=True):
    if keypress_detection_mode:
        while True:
            # k = keyboard.read_key(True) #True prevents keypress from propagating - doesn't work
            k = msvcrt.getch()
            k = k.decode('utf-8')
            if k in key_d:
                return key_d[k]
    
    while True:
        k = input('Enter your selection: ')
        if k in key_d:
            return key_d[k]