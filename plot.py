import matplotlib.pyplot as plt
from time import ctime


def plot_loss(data, title, current):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.plot(data)

    current = current.replace(' ', '-').replace(':', '-')
    fig.savefig(f'./results/{title}_{current}.jpg')