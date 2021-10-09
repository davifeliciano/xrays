from math import sin
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

LIGHT_SPEED = 299792458
CRYSTAL_LIF = 201.4e-12
CRYSTAL_KBR = 329.5e-12
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


# Importing files and sorting data
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

axes[-1].set(xlabel=r"$2\theta$ (\si{\degree})")

# Params to use in find_peaks for each dataset
fp_params = (
    {"height": 30.0, "threshold": None, "width": None},
    {"height": None, "threshold": 0.3, "width": 5},
    {"height": None, "threshold": 2.0, "width": None},
)

# Adding correction to the angles based on the result of angular.py
correction_df = pd.read_csv("results/angular_result.csv", decimal=",")
angle_stderr = 0.5

for ax, crystal, aperture, voltage, df, params in zip(
    axes, crystals, apertures, voltages, dfs, fp_params
):

    ax.set(
        title=f"Cristal de {crystal}, {aperture}mm de abertura, {voltage} kV",
        ylabel="Impulsos",
    )

    if aperture == 2.0:
        correction = correction_df.loc[0, "center"]
    if aperture == 5.0:
        correction = correction_df.loc[1, "center"]

    x_data = gaussian_filter1d(df.loc[:, "Angle"], sigma=2) - correction
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

    # Computing physical variables of interest
    k_alpha_angles_stderr = np.full(k_alpha_angles.size, angle_stderr)
    k_beta_angles_stderr = np.full(k_beta_angles.size, angle_stderr)
    k_alpha_bragg = k_alpha_angles / 2
    k_beta_bragg = k_beta_angles / 2
    k_alpha_bragg_stderr = k_alpha_angles_stderr
    k_beta_bragg_stderr = k_beta_angles_stderr
    k_alpha_bragg_rad = np.radians(k_alpha_bragg)
    k_beta_bragg_rad = np.radians(k_beta_bragg)
    k_alpha_bragg_rad_stderr = np.radians(k_alpha_bragg_stderr)
    k_beta_bragg_rad_stderr = np.radians(k_beta_bragg_stderr)

    one_over_sin_k_alpha = 1 / np.sin(k_alpha_bragg_rad)
    one_over_sin_k_beta = 1 / np.sin(k_beta_bragg_rad)

    one_over_sin_k_alpha_stderr = (
        k_alpha_bragg_rad_stderr
        * np.cos(k_alpha_bragg_rad)
        / np.sin(k_alpha_bragg_rad) ** 2
    )
    one_over_sin_k_beta_stderr = (
        k_beta_bragg_rad_stderr
        * np.cos(k_beta_bragg_rad)
        / np.sin(k_beta_bragg_rad) ** 2
    )

    if crystal == "LiF":
        CRYSTAL = CRYSTAL_LIF
    if crystal == "KBr":
        CRYSTAL = CRYSTAL_KBR

    k_alpha_energies_js = [
        0.5 * PLANCK_JS * LIGHT_SPEED * one_over_sin * (i + 1) / CRYSTAL
        for i, one_over_sin in enumerate(one_over_sin_k_alpha)
    ]
    k_alpha_energies_js_stderr = [
        0.5 * PLANCK_JS * LIGHT_SPEED * one_over_sin * (i + 1) / CRYSTAL
        for i, one_over_sin in enumerate(one_over_sin_k_alpha_stderr)
    ]

    k_alpha_energies_kev = [
        5e-4 * PLANCK_EV * LIGHT_SPEED * one_over_sin * (i + 1) / CRYSTAL
        for i, one_over_sin in enumerate(one_over_sin_k_alpha)
    ]
    k_alpha_energies_kev_stderr = [
        5e-4 * PLANCK_EV * LIGHT_SPEED * one_over_sin * (i + 1) / CRYSTAL
        for i, one_over_sin in enumerate(one_over_sin_k_alpha_stderr)
    ]

    k_beta_energies_js = [
        0.5 * PLANCK_JS * LIGHT_SPEED * one_over_sin * (i + 1) / CRYSTAL
        for i, one_over_sin in enumerate(one_over_sin_k_beta)
    ]
    k_beta_energies_js_stderr = [
        0.5 * PLANCK_JS * LIGHT_SPEED * one_over_sin * (i + 1) / CRYSTAL
        for i, one_over_sin in enumerate(one_over_sin_k_beta_stderr)
    ]

    k_beta_energies_kev = [
        5e-4 * PLANCK_EV * LIGHT_SPEED * one_over_sin * (i + 1) / CRYSTAL
        for i, one_over_sin in enumerate(one_over_sin_k_beta)
    ]
    k_beta_energies_kev_stderr = [
        5e-4 * PLANCK_EV * LIGHT_SPEED * one_over_sin * (i + 1) / CRYSTAL
        for i, one_over_sin in enumerate(one_over_sin_k_beta_stderr)
    ]

    ax.set(ylim=ylim)
    ax.legend(loc="upper left")

    # Puting data of interest in a DataFrame to export as a csv
    result_df = pd.DataFrame(
        [
            k_beta_angles,
            k_beta_angles_stderr,
            k_beta_bragg,
            k_beta_bragg_stderr,
            k_beta_energies_js,
            k_beta_energies_js_stderr,
            k_beta_energies_kev,
            k_beta_energies_kev_stderr,
            k_alpha_angles,
            k_alpha_angles_stderr,
            k_alpha_bragg,
            k_alpha_bragg_stderr,
            k_alpha_energies_js,
            k_alpha_energies_js_stderr,
            k_alpha_energies_kev,
            k_alpha_energies_kev_stderr,
        ],
        index=(
            "k_beta_angles",
            "k_beta_angles_stderr",
            "k_beta_bragg",
            "k_beta_bragg_stderr",
            "k_beta_energies_js",
            "k_beta_energies_js_stderr",
            "k_beta_energies_kev",
            "k_beta_energies_kev_stderr",
            "k_alpha_angles",
            "k_alpha_angles_stderr",
            "k_alpha_bragg",
            "k_alpha_bragg_stderr",
            "k_alpha_energies_js",
            "k_alpha_energies_js_stderr",
            "k_alpha_energies_kev",
            "k_alpha_energies_kev_stderr",
        ),
    )

    # Exporting it to results directory
    result_df.transpose().to_csv(
        f"results/spectre_{crystal}_{aperture}mm_{voltage}kV.csv",
        decimal=",",
        index=None,
    )

fig.savefig("plots/spectrum.png", dpi=300)
plt.show()
