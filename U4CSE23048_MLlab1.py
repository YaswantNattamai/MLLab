import math
import random
import statistics


def countVC1(text):
    vow = 0
    con = 0
    for i in text:
        if i in 'AEIOUaeiou':
            vow += 1
        else:
            con += 1
    return vow, con


def matrixMultiplier2(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return "Multiplication not possible. INVALID"

    rows1 = len(mat1)
    rows2 = len(mat2)
    cols1 = len(mat1[0])
    cols2 = len(mat2[0])

    op = [[0 for _ in range(cols2)] for _ in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                op[i][j] += mat1[i][k] * mat2[k][j]

    return op


def common3(nums1, nums2):
    common = []
    if len(nums1) > len(nums2):
        for i in nums1:
            if i in nums2 and i not in common:
                common.append(i)
    else:
        for i in nums2:
            if i in nums1 and i not in common:
                common.append(i)
    return common


def transpose4(mat1):
    rows = len(mat1)
    cols = len(mat1[0])
    op = [[0 for _ in range(rows)] for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            op[c][r] = mat1[r][c]
    return op


def randomStatistics5(n):
    op = [random.randrange(100, 150) for _ in range(n)]
    op.sort()
    mean1 = sum(op) / n
    mode1 = statistics.mode(op)
    if n % 2 == 1:
        median1 = op[n // 2]
    else:
        median1 = (op[n // 2 - 1] + op[n // 2]) / 2
    return mean1, mode1, median1



print("Lab Question 1: Vowel and Consonant Counter")
text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
vowels, consonants = countVC1(text)
print("Vowels in the text:", vowels)
print("Consonants in the text:", consonants)
print("\n" + "_" * 60 + "\n")

print("Lab Question 2: Matrix Multiplication")
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
result = matrixMultiplier2(a, b)
print("Matrix A:", a)
print("Matrix B:", b)
print("Result of multiplication:")
for row in result:
    print(row)
print("\n" + "_" * 60 + "\n")

print("Lab Question 3: Common Elements Finder")
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
print("List 1:", list1)
print("List 2:", list2)
print("Common elements:", common3(list1, list2))
print("\n" + "_" * 60 + "\n")

print("Lab Question 4: Matrix Transpose")
mat = [[1, 2, 3], [4, 5, 6]]
print("Original Matrix:")
for row in mat:
    print(row)
transposed = transpose4(mat)
print("Transposed Matrix:")
for row in transposed:
    print(row)
print("\n" + "_" * 60 + "\n")

print("Lab Question 5: Random Statistics")
mean1, mode1, median1 = randomStatistics5(11)
print("Mean:", mean1)
print("Mode:", mode1)
print("Median:", median1)
