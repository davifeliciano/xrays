import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{siunitx}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "figure.figsize": [10.0, 5.0],
    }
)

files = glob.glob("data/data1/Perfil angular*.dat")
apertures = [int(re.search("angular (.+?)mm.dat", file).group(1)) for file in files]
dfs = [
    pd.read_csv(file, sep="\s+", skiprows=(1, 2), decimal=",", encoding="ISO-8859-1")
    for file in files
]

fig, axes = plt.subplots(1, 2)


def gaussian(x, a, center, width):
    return a * np.exp(-(((x - center) / width) ** 2) / 2)


center = []
center_stderr = []
width = []
width_stderr = []

for aperture, ax, df in zip(apertures, axes, dfs):
    x_data = df.loc[:, "Angle"]
    y_data = df.loc[:, "Impulses"]

    ax.set(
        title=f"Abertura de {aperture}mm",
        xlim=(x_data[0], x_data.iloc[-1]),
        xlabel=r"$\theta$ (\si{\degree})",
        ylabel="Impulsos",
    )

    params, params_cov = curve_fit(gaussian, x_data, y_data)
    params_sdterr = np.sqrt(np.diag(params_cov))

    center.append(params[1])
    center_stderr.append(params_sdterr[1])
    width.append(abs(params[2]))
    width_stderr.append(params_sdterr[2])

    x = np.linspace(x_data.iloc[0], x_data.iloc[-1], 500)

    ax.plot(x, gaussian(x, *params), label="Ajuste Gaussiano", zorder=-1)
    ax.plot(x_data, y_data, "o", ms=3.5, label=f"Medidas")

    ax.legend()
    ax.grid(linestyle=":")

result_df = pd.DataFrame(
    np.array([apertures, center, center_stderr, width, width_stderr]).T,
    columns=("aperture", "center", "center_stderr", "width", "width_stderr"),
)

result_df.to_csv("results/angular_result.csv", decimal=",", index=None)

fig.savefig("plots/angular.png", dpi=300)
plt.show()
