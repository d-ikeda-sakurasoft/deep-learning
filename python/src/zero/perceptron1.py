def AND(x1, x2):
    w1, w2, t = 0.5, 0.5, 0.7
    return int((x1 * w1 + x2 * w2) > t)

def NAND(x1, x2):
    w1, w2, t = -0.5, -0.5, -0.7
    return int((x1 * w1 + x2 * w2) > t)

def OR(x1, x2):
    w1, w2, t = 0.5, 0.5, 0.4
    return int((x1 * w1 + x2 * w2) > t)

print("AND")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

print("NAND")
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))

print("OR")
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))
