#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
fig, a = plt.subplots()
plt.ylim((0, 80))
persons = ["Farrah", "Fred", "Felicia"]
plt.ylabel('Quantity of Fruit')
plt.title("Number of Fruit per Person")
colors = ['r', 'yellow', '#ff8000', '#ffe5b4']
f_type = ["apples", "bananas", "oranges", "peaches"]
for i in range(fruit.shape[0]):
    a.bar(persons, fruit[i], bottom=np.sum(fruit[:i], axis=0),
          width=0.5, color=colors[i % len(colors)],
          label=f_type[i % len(f_type)])
    a.legend()
plt.show()
