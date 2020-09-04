import sys


def wc(filename):
    count = 0
    with open(filename) as f:
        for _ in f:
            count += 1

    return count

def process_file(filename):
    count = wc(filename)
    print(f'File: {filename} has {count} lines')

def _main():
    process_file(sys.argv[1])

if __name__ == '__main__':
    _main()
