from math import sin
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

LIGHT_SPEED = 299792458
CRYSTAL = 201.4e-12
PLANCK_JS = 6.62607004e-34
PLANCK_EV = 4.13566769e-15
RYDBERG = 3.28989e15

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{siunitx}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "figure.figsize": [5.0, 10.0],
    }
)


def get_filename(path):
    return path.split("\\")[-1].split("/")[-1]


files = glob.glob("data/data1/Tubo*.dat")
crystals = [get_filename(file).split(" ")[4] for file in files]
apertures = [
    int(re.search(f"{crystals[i]} (.+?)mm ", file).group(1))
    for i, file in enumerate(files)
]

voltages = [float(re.search("mm (.+?) keV", file).group(1)) for file in files]
dfs = [
    pd.read_csv(file, sep="\s+", skiprows=(1, 2), decimal=",", encoding="ISO-8859-1")
    for file in files
]

fig, axes = plt.subplots(3, 1)

axes[-1].set(xlabel=r"$\theta$ (\si{\degree})")

fp_params = (
    {"height": None, "threshold": 0.3, "width": 5},
    {"height": 30.0, "threshold": None, "width": None},
    {"height": None, "threshold": 2.0, "width": None},
)

for ax, crystal, aperture, voltage, df, params in zip(
    axes, crystals, apertures, voltages, dfs, fp_params
):

    ax.set(
        title=f"Cristal de {crystal}, {aperture}mm de abertura, {voltage} kV",
        ylabel="Impulsos",
    )

    x_data = gaussian_filter1d(df.loc[:, "Angle"], sigma=2)
    y_data = gaussian_filter1d(df.loc[:, "Impulses"], sigma=2)

    ax.plot(x_data, y_data)

    ylim = ymin, ymax = ax.get_ylim()
    peak_indexes = list(
        find_peaks(
            y_data,
            height=params["height"],
            threshold=params["threshold"],
            width=params["width"],
        )[0]
    )

    k_alpha_indexes = [peak for i, peak in enumerate(peak_indexes) if i % 2 != 0]
    k_beta_indexes = [peak for i, peak in enumerate(peak_indexes) if i % 2 == 0]
    k_alpha_angles = x_data[k_alpha_indexes]
    k_beta_angles = x_data[k_beta_indexes]
    one_over_sin_k_alpha = 1 / np.sin(np.radians(k_alpha_angles))
    one_over_sin_k_beta = 1 / np.sin(np.radians(k_beta_angles))

    k_alpha_energies_js = [
        0.5 * PLANCK_JS * LIGHT_SPEED * one_over_sin / ((i + 1) * CRYSTAL)
        for i, one_over_sin in enumerate(one_over_sin_k_alpha)
    ]
    k_alpha_energies_kev = [
        5e-4 * PLANCK_EV * LIGHT_SPEED * one_over_sin / ((i + 1) * CRYSTAL)
        for i, one_over_sin in enumerate(one_over_sin_k_alpha)
    ]
    k_beta_energies_js = [
        0.5 * PLANCK_JS * LIGHT_SPEED * one_over_sin / ((i + 1) * CRYSTAL)
        for i, one_over_sin in enumerate(one_over_sin_k_beta)
    ]
    k_beta_energies_kev = [
        5e-4 * PLANCK_EV * LIGHT_SPEED * one_over_sin / ((i + 1) * CRYSTAL)
        for i, one_over_sin in enumerate(one_over_sin_k_beta)
    ]

    ax.vlines(
        k_alpha_angles,
        ymin,
        ymax,
        color="red",
        linewidth=0.5,
        label=r"$K_{\alpha}$",
    )

    ax.vlines(
        k_beta_angles,
        ymin,
        ymax,
        color="orange",
        linewidth=0.5,
        label=r"$K_{\beta}$",
    )

    ax.set(ylim=ylim)
    ax.legend(loc="upper left")

    result_df = pd.DataFrame(
        [
            k_beta_angles,
            k_beta_energies_js,
            k_beta_energies_kev,
            k_alpha_angles,
            k_alpha_energies_js,
            k_alpha_energies_kev,
        ],
        index=(
            "k_beta_angles",
            "k_beta_energies_js",
            "k_beta_energies_kev",
            "k_alpha_angles",
            "k_alpha_energies_js",
            "k_alpha_energies_kev",
        ),
    )

    result_df.transpose().to_csv(
        f"results/spectre_{crystal}_{aperture}mm_{voltage}kV.csv",
        decimal=",",
        index=None,
    )

fig.savefig("plots/spectrum.png", dpi=300)
plt.show()
