import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

ELECTRON = 1.60217662e-19
LIGHT_SPEED = 299792458
CRYSTAL = 201.4e-12
PLANCK = 6.62607004e-34

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{siunitx}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "figure.figsize": [10.0, 5.0],
    }
)

# Importing files
files = glob.glob("data/data1/X-ray*.dat")
voltages = [float(re.search("angle (.+?) keV", file).group(1)) for file in files]
dfs = [
    pd.read_csv(file, sep="\s+", skiprows=(1, 2), decimal=",", encoding="ISO-8859-1")
    for file in files
]

# Sorting dfs by corresponding voltage
dfs = [df for _, df in sorted(zip(voltages, dfs), key=lambda pair: pair[0])]
voltages.sort()

fig, ax = plt.subplots(1, 2)

# First plot: impulses in terms of angles
ax[0].set(xlabel=r"$\theta$ (\si{\degree})", ylabel="Impulsos")

# Adding correction to the angles based on the result of angular.py
correction_df = pd.read_csv("results/angular_result.csv", decimal=",")
correction = correction_df.loc[0, "center"]

slopes = []
slopes_stderr = []
intercepts = []
intercepts_stderr = []
rvalues = []
roots = []
roots_stderr = []

for voltage, df in zip(voltages, dfs):
    x_data = df.loc[:, "Angle"] + correction
    y_data = df.loc[:, "Impulses"] + correction

    x_filtered = gaussian_filter1d(x_data, sigma=3)
    y_filtered = gaussian_filter1d(y_data, sigma=3)
    plot = ax[0].plot(x_filtered, y_filtered, label=rf"{voltage} \si{{kV}}")
    color = plot[-1].get_color()

    lenght = len(x_filtered)
    start = int(3 * lenght / 5)
    stop = int(4 * lenght / 5)

    reg = linregress(x_filtered[start:stop], y_filtered[start:stop])

    slope = reg.slope
    slope_stderr = reg.stderr
    intercept = reg.intercept
    intercept_stderr = reg.intercept_stderr
    rvalue = reg.rvalue
    root = -intercept / slope
    root_stderr = root * np.sqrt(
        abs(reg.stderr / slope) ** 2 + abs(reg.intercept_stderr / intercept) ** 2
    )

    slopes.append(slope)
    slopes_stderr.append(slope_stderr)
    intercepts.append(intercept)
    intercepts_stderr.append(intercept_stderr)
    rvalues.append(rvalue)
    roots.append(root)
    roots_stderr.append(root_stderr)

    x = np.linspace(root, x_filtered[-1])
    ax[0].plot(x, slope * x + intercept, ":", color=color)

# df with the regression results to export as csv
reg_df = pd.DataFrame(
    [
        voltages,
        slopes,
        slopes_stderr,
        intercepts,
        intercepts_stderr,
        rvalues,
        roots,
        roots_stderr,
    ],
    index=(
        "voltage",
        "slope",
        "slope_stderr",
        "intercept",
        "intercept_stderr",
        "rvalue",
        "root",
        "root_stderr",
    ),
)

reg_df.transpose().to_csv(f"results/duane_hunt_result.csv", decimal=",", index=None)

ax[0].legend()
ax[0].grid(linestyle=":")

# Second plot: voltages in terms of 1 / sin(angle)
ax[1].set(xlabel=r"$1 / \sin\theta_{min}$", ylabel=r"$U$ (\si{kV})")

roots = np.radians(np.array(roots))
one_over_sin = 1 / np.sin(roots)
one_over_sin_stderr = abs(1 / np.sin(roots) / np.tan(roots)) * (
    np.pi * np.array(roots_stderr) / 180
)

reg = linregress(one_over_sin[:-1], voltages[:-1])

slope = reg.slope
slope_stderr = reg.stderr
intercept = reg.intercept
intercept_stderr = reg.intercept_stderr
rvalue = reg.rvalue

# df with the regression results to export as csv
reg_df = pd.DataFrame(
    [
        [
            slope,
            slope_stderr,
            intercept,
            intercept_stderr,
            rvalue,
        ]
    ],
    columns=(
        "slope",
        "slope_stderr",
        "intercept",
        "intercept_stderr",
        "rvalue",
    ),
)

reg_df.to_csv(f"results/duane_hunt_reg_result.csv", decimal=",", index=None)

ax[1].errorbar(
    one_over_sin[:-1],
    voltages[:-1],
    xerr=one_over_sin_stderr[:-1],
    fmt="o",
    ms=3.5,
    label="Pontos Considerados no Ajuste",
)

ax[1].errorbar(
    one_over_sin[-1],
    voltages[-1],
    xerr=one_over_sin_stderr[-1],
    color="red",
    fmt="o",
    ms=3.5,
    label="Pontos Desprezados no Ajuste",
)

xlim = xmin, xmax = ax[1].get_xlim()
x = np.linspace(xmin, xmax)

ax[1].plot(x, slope * x + intercept, color="orange", label="Ajuste Linear", zorder=-1)

ax[1].set(xlim=xlim)

expected_slope = PLANCK * LIGHT_SPEED / (2000 * ELECTRON * CRYSTAL)
expected_voltages = expected_slope * one_over_sin
expected_voltages_stderr = expected_slope * one_over_sin_stderr

ax[1].plot(x, expected_slope * x, color="green", label="Reta Esperada", zorder=-1)
ax[1].errorbar(
    one_over_sin,
    expected_voltages,
    yerr=expected_voltages_stderr,
    color="purple",
    fmt="o",
    ms=3.5,
    label="Tensões Esperadas",
)

# df with expected voltages and standard errors
expected_voltages_df = pd.DataFrame(
    np.array([voltages, expected_voltages, expected_voltages_stderr]).T,
    columns=("value", "expected_value", "stderr"),
)

expected_voltages_df.to_csv("results/expected_voltages.csv", decimal=",", index=None)

ax[1].legend()
ax[1].grid(linestyle=":")

fig.savefig("plots/duane_hunt.png", dpi=300)
plt.show()
