import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline

LIGHT_SPEED = 299792458
CRYSTAL = 201.4e-12
PLANCK_JS = 6.62607004e-34
PLANCK_EV = 4.13566769e-15
RYDBERG = 3.28989e15

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{siunitx}\usepackage{mathtools}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "figure.figsize": [12.0, 18.0],
    }
)


def get_filename(path):
    return path.split("\\")[-1].split("/")[-1]


# Importing files
files = glob.glob("data/data2/Moseley Borda K/data_selection_1/*.dat")
elements = [get_filename(file).split()[0] for file in files]
atomic_number = {"Zn": 30, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Rb": 37, "Sr": 38}
dfs = [
    pd.read_csv(file, sep="\s+", skiprows=(1, 2), decimal=",", encoding="ISO-8859-1")
    for file in files
]

# Sorting dfs and elements by atomic number
sorted_dfs = []
sorted_elems = []
for df, element in sorted(zip(dfs, elements), key=lambda pair: atomic_number[pair[1]]):
    sorted_dfs.append(df)
    sorted_elems.append(element)

dfs = sorted_dfs
elements = sorted_elems

# First plot: k absorption edge of various elements
fig, axes = plt.subplots(4, 2, sharex=True, sharey=True)
fig.delaxes(axes[3, 1])

# Setting labels for the axes
axes[-1, 0].set(xlabel=r"$\theta$ (\si{\degree})")
axes[-2, 1].set(xlabel=r"$\theta$ (\si{\degree})")
for ax in axes[:, 0].flat:
    ax.set(ylabel="Impulsos")

# Range of angles inside which is expected an absorption edge
abs_range = {
    "Zn": (18.0, 19.0),
    "Ge": (15.5, 16.5),
    "As": (14.5, 15.5),
    "Se": (13.5, 14.5),
    "Br": (12.5, 13.5),
    "Rb": (11.0, 12.0),
    "Sr": (10.5, 11.5),
}


def get_inflection(x_data, y_data):
    """Return the x values of inflection points given x and y numpy arrays"""
    # Create an spline that fits the data
    y_spl = UnivariateSpline(x_data, y_data)
    # Create an spline that represents its second derivative
    y_spl_sd = y_spl.derivative(n=2)
    domain = np.arange(5, 20, 0.01).flat
    # Return a list of values of x for which y_spl_sd == 0
    return [
        x
        for i, x in enumerate(domain[:-1])
        if np.sign(y_spl_sd(x)) == -np.sign(y_spl_sd(domain[i + 1]))
    ]


# Color generator to use in plots
def color_gen(lenght):
    for i in range(lenght):
        yield f"C{i}"


color = color_gen(len(elements))
ylim = ymin, ymax = (-100, 3500)

# Adding correction to the angles based on the result of angular.py
correction_df = pd.read_csv("results/angular_result.csv", decimal=",")
correction = correction_df.loc[1, "center"]
angle_stderr = correction_df.loc[1, "center_stderr"]

k_abs_edges = []

for ax, df, element in zip(axes.flat, dfs, elements):
    x_data = df.loc[:, "Angle"].to_numpy() / 2 + correction
    y_data = df.loc[:, "Impulses"].to_numpy()

    x_data_filtered = gaussian_filter1d(x_data, sigma=1.5)
    y_data_filtered = gaussian_filter1d(y_data, sigma=1.5)

    ax.tick_params(reset=True)

    next_color = next(color)
    ax.plot(
        x_data,
        y_data,
        color=next_color,
        label=rf"$\prescript{{}}{{{atomic_number[element]}}}{{\mathrm{{{element}}}}}$",
    )
    ax.plot(x_data_filtered, y_data_filtered, "--", color=next_color, linewidth=0.7)

    inflection_angles = get_inflection(x_data_filtered, y_data_filtered)

    found = False
    for angle in inflection_angles:
        if angle >= abs_range[element][0] and angle <= abs_range[element][1]:
            found = True
            k_abs_edge = angle
            k_abs_edges.append(k_abs_edge)
    if not found:
        raise ValueError(
            f"No inflection point found in range ({abs_range[element][0]}, {abs_range[element][1]}) for element {element}"
        )

    ax.vlines(
        k_abs_edge,
        ymin,
        ymax,
        color="red",
        linewidth=0.5,
        label=rf"$\theta = {k_abs_edge:.1f}\si{{\degree}}$",
    )

    ax.set(ylim=ylim)
    ax.legend(loc="upper left")

# Second plot: Moseley diagram (atomic number in terms of sqrt of energy)

# Computing physical variables of interest
k_abs_edges_stderr = np.full(len(k_abs_edges), angle_stderr)
k_abs_edges_rad = np.radians(k_abs_edges)
k_abs_edges_rad_stderr = np.radians(angle_stderr)
one_over_sin = 1 / np.sin(k_abs_edges_rad)
one_over_sin_stderr = (
    np.radians(angle_stderr) * np.cos(k_abs_edges_rad) / np.sin(k_abs_edges_rad) ** 2
)

energies_js = 0.5 * PLANCK_JS * LIGHT_SPEED * one_over_sin / CRYSTAL
energies_js_stderr = 0.5 * PLANCK_JS * LIGHT_SPEED * one_over_sin_stderr / CRYSTAL
energies_kev = 5e-4 * PLANCK_EV * LIGHT_SPEED * one_over_sin / CRYSTAL
energies_kev_stderr = 5e-4 * PLANCK_EV * LIGHT_SPEED * one_over_sin_stderr / CRYSTAL
sqrt_energies_js = np.sqrt(energies_js)
sqrt_energies_js_stderr = 0.5 * energies_js_stderr / np.sqrt(energies_js)
sqrt_energies_kev = np.sqrt(energies_kev)
sqrt_energies_kev_stderr = 0.5 * energies_kev_stderr / np.sqrt(energies_kev)

zs = [atomic_number[element] for element in elements]

# df with the data of the plot to export as a csv
result_df = pd.DataFrame(
    [
        elements,
        zs,
        k_abs_edges,
        k_abs_edges_stderr,
        energies_js,
        energies_js_stderr,
        energies_kev,
        energies_kev_stderr,
        sqrt_energies_js,
        sqrt_energies_js_stderr,
        sqrt_energies_kev,
        sqrt_energies_kev_stderr,
    ],
    index=(
        "element",
        "z",
        "k_abs_edges",
        "k_abs_edges_stderr",
        "energies_js",
        "energies_js_stderr",
        "energies_kev",
        "energies_kev_stderr",
        "sqrt_energies_js",
        "sqrt_energies_js_stderr",
        "sqrt_energies_kev",
        "sqrt_energies_kev_stderr",
    ),
)

# Transposing and changing dtype to float on suitable columns
result_df = result_df.transpose()
for column in result_df.columns:
    if column != "element" and column != "z":
        result_df[column] = pd.to_numeric(result_df[column])

result_df.to_csv("results/k_abs_edges.csv", decimal=",", index=None)

# Linear regression of the data
reg = linregress(sqrt_energies_js, zs)

slope = reg.slope
slope_stderr = reg.stderr
intercept = reg.intercept
intercept_stderr = reg.intercept_stderr
rvalue = reg.rvalue
expected_slope = 1 / np.sqrt(RYDBERG * PLANCK_JS)

# df with the regression results to export as csv
reg_df = pd.DataFrame(
    [[slope, slope_stderr, expected_slope, intercept, intercept_stderr, rvalue]],
    columns=(
        "slope",
        "slope_stderr",
        "expected_slope",
        "intercept",
        "intercept_stderr",
        "rvalue",
    ),
)

reg_df.to_csv("results/moseley_reg_result.csv", decimal=",", index=None)

fig_moseley, ax_moseley = plt.subplots(figsize=(5, 4))

ax_moseley.set(xlabel=r"$\sqrt{E} \; (\si{Js})$", ylabel="Z")

ax_moseley.errorbar(
    sqrt_energies_js,
    zs,
    xerr=sqrt_energies_js_stderr,
    fmt="none",
    color="blue",
    ms=3.5,
    label="Dados Considerados no Ajuste",
)

xlim = xmin, xmax = ax_moseley.get_xlim()
x = np.linspace(xmin, xmax)

ax_moseley.plot(
    x, slope * x + intercept, color="orange", label="Ajuste Linear", zorder=-1
)

ax_moseley.set(xlim=xlim)
ax_moseley.legend()
ax_moseley.grid(linestyle=":")

fig.savefig("plots/k_abs_edges.png", dpi=300)
fig_moseley.savefig("plots/moseley_diagram.png", dpi=300)
plt.show()
