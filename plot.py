import matplotlib.pyplot as plt


def plot_loss(data, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.plot(data)
    fig.savefig(f'./results/{title}.jpg')