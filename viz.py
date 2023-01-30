# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
import logging
import traceback
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import umap

# import librosa
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from filters import bandpassSOS, drop_back


# GENERAL EXTRACTED FEATURE VIZUALIZATION
def plot_tSNE(
    data: np.ndarray, comp_num: int, title="tSNE plot", multiple: bool = False
):
    """
    returns plot for the tSNE of the data
    Input data must be 2dim vector
    This function is very sensitive to the perplexity
    argument passed:
    """

    if not multiple:

        flattened = np.array([col.flatten() for col in data.T])

    else:

        flattened = list()

        for spec in data:
            for col in spec.T:
                flattened.append(col.flatten())

    flattened = np.array(flattened)
    X_embedding = TSNE(n_components=comp_num)

    X_embedding_transformed = X_embedding.fit_transform(flattened)  # has to be dim 2

    x, y = X_embedding_transformed.T

    figure = plt.figure()
    plt.scatter(x, y, color="r", s=50, alpha=0.8)

    labels = list(range(len(x)))

    norm = mpl.colors.Normalize(vmin=min(labels), vmax=max(labels))
    cmap = "Purples"
    figure = plt.figure(figsize=(6, 6), dpi=200)
    ax = figure.gca()
    ax.scatter(x, y, c=labels, s=50, alpha=1, cmap=cmap, norm=norm)
    ax.grid()

    ax.set_title(title)
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    figure.show()

    return figure


def plot_UMAP(data: np.ndarray, title="UMAP plot", multiple: bool = False) -> list:
    """
    Returns a plot object of a UMAP projection from input data
    """

    if not multiple:

        flattened = np.array([col.flatten() for col in data.T])

    else:

        flattened = list()

        for spec in data:
            for col in spec.T:
                flattened.append(col.flatten())

    flattened = np.array(flattened)

    model = umap.UMAP()

    model_transformed = model.fit_transform(flattened)

    x, y = model_transformed.T

    labels = list(range(len(x)))

    norm = mpl.colors.Normalize(vmin=min(labels), vmax=max(labels))
    cmap = "Greens"
    figure = plt.figure(figsize=(6, 6), dpi=200)
    ax = figure.gca()
    ax.scatter(x, y, c=labels, s=50, alpha=1, cmap=cmap, norm=norm)
    ax.grid()
    ax.set_title(title)
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    figure.show()

    return figure


def plot_hessian(data: np.ndarray) -> list:
    """
    Returns a plot object of a LLE hessian from the input data
    """

    n_components = 2

    n_neighbors = n_components * (n_components + 3) // 2
    n_neighbors = n_neighbors + 60

    flattened = np.array([row.flatten() for row in data])

    model = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        method="hessian",
        eigen_solver="dense",
    )

    model_transformed = model.fit_transform(flattened)

    x, y = model_transformed.T

    figure = plt.figure()
    plt.scatter(x, y, color="r", s=50, alpha=0.8)

    labels = list(range(len(x)))

    norm = mpl.colors.Normalize(vmin=min(labels), vmax=max(labels))
    cmap = "Purples"
    figure = plt.figure()
    ax = figure.gca()
    ax.scatter(x, y, c=labels, s=50, alpha=1, cmap=cmap, norm=norm)
    ax.grid()
    ax.set_title("Hessian Locally Linear Embedding")
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    figure.show()

    return figure


def plot_specs(num_row: int, num_col: int, sr: int, signal: np.ndarray, graph=False):

    """
    Creates a matrix of spectrogram subplots over the signal data
    with num_row rows and num_col columns.

    sr, sample rate, is needed to determine the slide across the spectrograms.

    By default the graph is off.
    """

    num_subplots = int(num_row * num_col)

    fig, ax = plt.subplots(num_row, num_col, sharex=True, sharey=True)

    spec_data = list()
    wind = 0

    for colm in range(num_col):
        for row in range(num_row):
            wind += 1
            start = sr * wind
            stop = start + sr

            fft_data = librosa.stft(signal[start:stop])
            mag_data = librosa.power_to_db(np.abs(fft_data), ref=np.max)

            spec_data.append(mag_data)

            fig = librosa.display.specshow(
                mag_data, y_axis="hz", x_axis="time", sr=sr, ax=ax[row, colm]
            )
            ax[row, colm].set_title(f"time range {start//sr} sec to {stop//sr} sec")
            # ax[row,colm].set_ylim(2000,8000)

    plt.show()

    return spec_data


def view_sec(time: int, signal: np.ndarray, sr: int) -> int:
    """
    Funciton written to view filtered sections of audio data to check behavior of order after filtering.
    Needs to be fixed becaus relies on old parsing algo and it's not correct.

    """

    # how do i get time from a spectrogram to make slices in the spectrogram according to time

    n_fft = sr // 20
    hop = n_fft // 2

    start = sr * time
    stop = start + sr
    view_range = signal[start:stop]

    filtered_view_range = bandpassSOS(signal, 1, sr, 100000, 18000)

    fig, ax = plt.subplots(1, 1)
    fft_data = librosa.stft(filtered_view_range, n_fft=n_fft, hop_length=hop)
    mag_data = librosa.power_to_db(np.abs(fft_data), ref=np.max)

    black = drop_back(mag_data, 25)

    duration = librosa.get_duration(
        S=black, sr=sr, n_fft=n_fft, hop_length=hop
    )  # duration of recording in seconds
    duration = duration * 1000  # sec -> ms conversion
    num_sections = int(duration // 1)  # as miliseconds

    fig = librosa.display.specshow(
        black, y_axis="hz", x_axis="time", sr=sr, n_fft=n_fft, hop_length=hop, ax=ax
    )
    ax.set(title=f"1 second Power Spectrogram of rat call at second {time}.")
    plt.ylim(18000, 100000)
    plt.colorbar(fig, ax=ax, format="%+2.f dB")
    plt.show()
    return 0


def plot_Isomap2D(spectrogram: np.ndarray, n_neighbors=5, radius=None, graph=False):

    flattened = np.array([row.flatten() for row in spectrogram])

    assert flattened.ndim == 2, "Input tensor must be of dimension 2"

    try:

        isomap_model = Isomap(n_neighbors=n_neighbors, radius=radius, n_components=2)

        isomap_model_transformed = isomap_model.fit_transform(
            flattened
        )  # has to be dim 2

        x, y = isomap_model_transformed.T
        dashape = isomap_model_transformed.shape
        print(f"shape is{dashape}")

        if graph:
            plt.scatter(x, y, color="g", s=50, alpha=0.8)
            plt.show()

            return 0

    except:
        logging.error(traceback.format_exc)


def draw_umap(
    *, data, n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", title=""
):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=data, s=100)
    plt.title(title, fontsize=18)


#### NERUAL NETWORK VIZUALIZATION FOR LATER
def plot_history(history):

    """
    Plots accuracy/loss for training/validation
    takes a history list which is the list turned from a
    after fitting a TF model.
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()
