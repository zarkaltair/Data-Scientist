def gcd(a, b):
    if b % a == 0 or a % b == 0:
        return min(a, b)
    else:
        return gcd(min(a, b), max(a, b) % min(a, b))


def main():
    a, b = map(int, input().split())
    print(gcd(a, b))


if __name__ == "__main__":
    main()
