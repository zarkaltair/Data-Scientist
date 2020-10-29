@contract
def matrix_multiply(a, b):
    """ Multiplies two matrices together.

        :param a: The first matrix. 2D array.
        :type a: array[MxN],M>0,N>0

        :param b: The second matrix. 2D array
                  of compatible dimensions.
        :type b: array[NxP], P>0

        :rtype: array[MxP]
    """
    return numpy.dot(a, b)
