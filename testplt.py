import matplotlib.pyplot as plt

with open("C:\\Users\\jo94s\\Github\\OptConquest\\fmed_evolution.txt") as f:
    lines = f.readlines()
    x_evo = range(len(lines))
    y_evo = [float(line.split()[0]) for line in lines]

plt.plot(y_evo)

plt.show()