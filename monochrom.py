import glob
from os import sep
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{siunitx}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "figure.figsize": [5.0, 10.0],
    }
)

files = glob.glob("data\data2\Monocromatização\Monocromatização*.dat")
monochrom_types = [file.split()[-1].split(".")[0] for file in files]

dfs = [
    pd.read_csv(
        file,
        sep="\s+",
        skiprows=(1, 2),
        decimal=",",
        encoding="ISO-8859-1",
    )
    for file in files
]

df_ref = pd.read_csv(
    "data\data2\Monocromatização\Tubo de Cobre 2mm 26 keV ref para filtragem.dat",
    sep="\s+",
    skiprows=(1, 2),
    decimal=",",
    encoding="ISO-8859-1",
)
dfs.append(df_ref)
monochrom_types.append("none")

fig, axes = plt.subplots(3, 1)

axes[-1].set(xlabel=r"$2\theta$ (\si{\degree})")

# Adding correction to the angles based on the result of angular.py
correction_df = pd.read_csv("results/angular_result.csv", decimal=",")
correction = correction_df.loc[0, "center"]

peak_angles = []

for ax, df, type in zip(axes, dfs, monochrom_types):
    x_data = gaussian_filter1d(df.loc[:, "Angle"], sigma=2) - correction
    y_data = gaussian_filter1d(df.loc[:, "Impulses"], sigma=2)

    x_data_ref = gaussian_filter1d(df_ref.loc[:, "Angle"], sigma=2)
    y_data_ref = gaussian_filter1d(df_ref.loc[:, "Impulses"], sigma=2)

    if type == "Ni":
        title = "Monocromatização com filtro de Ni"
    if type == "difração":
        title = "Monocromatização por difração"
    if type == "none":
        title = "Sem monocromatização"

    ax.set(title=title, ylabel="Impulsos")

    ax.plot(x_data, y_data)

    ylim = ymin, ymax = ax.get_ylim()
    peak_index = list(
        find_peaks(
            y_data,
            height=200,
        )[0]
    )

    peak_angle = x_data[peak_index][0]
    peak_angles.append(peak_angle)

    ax.vlines(
        peak_angle,
        ymin,
        ymax,
        color="red",
        linewidth=0.5,
        label=rf"$\theta = {peak_angle:.1f}\si{{\degree}}$",
    )

    ax.set(ylim=ylim)
    ax.legend()

result_df = pd.DataFrame([monochrom_types, peak_angles], index=("type", "peak_angles"))
result_df.transpose().to_csv("results/monochrom_result.csv", decimal=",", index=None)

fig.savefig("plots/monochrom.png", dpi=300)
plt.show()
