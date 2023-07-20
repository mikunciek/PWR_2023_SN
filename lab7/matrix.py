# calculate a confusion matrix
def confusion_matrix(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]

    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()

    for i, value in enumerate(unique):
        lookup[value] = i

    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return unique, matrix


# pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
    print('(A)' + maxlen(unique, ' ', -2) + '| ' + ' '.join(str(x) for x in unique))
    print('(P)' + maxlen(unique, ' ', -2) + '| ' + ''.join(['-' for _ in range(0, len(''.join(unique)) + 1)]))
    for i, x in enumerate(unique):
        print("%s | %s" % (pad(unique, str(x)),''.join(mpad(str(x), i, idx, matrix) + maxlen(unique, ' ', 0) for idx, x in enumerate(matrix[i]))))

#change style of view matrix
def pad(unique: set, x: str):
    ulen: int = len(max(map(str, unique), key=len))
    xlen = len(x)

    if xlen == ulen:
        return x

    www = x + ''.join([' ' for i in range(0, ulen-xlen)])
    return www

def mpad(x: str, i: int, j:int, matrix: list[list]):
    tmp = [x[j] for x in matrix ]
    return pad(set(tmp), x)

def maxlen(array, char, mod):
    xlen = len(max(map(str, array), key=len))
    return ''.join([char for _ in range(0, xlen + mod)])

# Test confusion matrix with text
actual1 = ["22", "11111", "22", "11111", "11111", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22"]
predicted1 = ["11111", "22", "22", "11111", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22", "22"]

# unique, matrix = confusion_matrix(actual, predicted)
unique1, matrix1 = confusion_matrix(actual1, predicted1)

print_confusion_matrix(unique1, matrix1)
# print_confusion_matrix(unique, matrix)
