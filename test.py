import numpy as np
import math
import datetime
import random

x, y, z = [False] * 3
print(x, y, z)

x = 5, 5
y = 10, 10

print(math.dist(x, y))

class Test:

    def __init__(self):
        pass

x = Test()
x.score = 5
print(x.score)
y = None
print(not y)

t = datetime.datetime.now()
x = 1
for i in range(100000):
    x += 1
t2 = float(str(datetime.datetime.now() - t)[-9:])
print(t2)

x = np.array([11.12345])
print(x.round(3))
print()

class Test:

    def __init__(self):
        self.x = 5
        self.y = 5

    def func1(self):
        self.x = 10

        def func2():
            self.y = 10
            return 10

        return 5

p = .1
for i in range(5):
    if random.random() < p:
        break
else:
    print('boop', i)
print('beep', i)


x = [1,2,3,4,5,6,7,8]
print(random.choices(x, k=3))
for i, j in zip([[]], [[]]):
    pass

x = tuple("5, 10, 20")
print(x)