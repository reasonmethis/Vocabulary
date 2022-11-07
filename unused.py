def flush_input():
    try:
        import msvcrt

        n = 0
        print(n, n, n)
        while msvcrt.kbhit():
            n += 1
            msvcrt.getch()
        print(n, n, n)
    except ImportError:
        import sys, termios

        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
