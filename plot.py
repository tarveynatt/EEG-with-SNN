import matplotlib.pyplot as plt
from time import ctime


def plot_graph(data, title, current):
    fig, ax = plt.subplots()
    ax.set_title(title)
    # ax.set_xlabel('epoch')
    ax.set_xlabel('T')
    ax.plot(data)

    current = current[4:-5].replace(' ', '_').replace(':', '-')
    fig.savefig(f'./results/{title}_{current}.jpg')