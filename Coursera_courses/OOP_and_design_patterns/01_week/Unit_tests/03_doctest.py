import stack


def is_braces_sequenct_correct(seq: str) -> bool:
    """
    Check correctness of braces sequence in statement
    >>> is_braces_sequenct_correct('() (())')
    True
    >>> is_braces_sequenct_correct('()[()]')
    True
    >>> is_braces_sequenct_correct(')')
    False
    >>> is_braces_sequenct_correct('[()')
    False
    >>> is_braces_sequenct_correct('[()])')
    False
    """
    correspondent = dict(zip('([{', ')]}'))
    for brace in seq:
        if brace in '([{':
            stack.push(brace)
            continue
        elif brace in ')]}':
            if stack.is_empty():
                return False
            left = stack.pop()
            if correspondent[left] != brace:
                return False

        return stack.is_empty()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
