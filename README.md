# ROMSA: Paleostress Analysis from Faults

**ROMSA** (Right Dihedra Method Stress Analysis) is a high-performance Python tool for determining paleostress orientations from fault-slip data.

It is a modern re-implementation of the original C++ program published by **Bruno Ciscato** (1994), based on the structural analysis method developed by **Lisle** (1987).

![ROMSA Output Plot](examples/example_plot.png)

## 📜 History & Evolution

* **1994 (Original):** Written in C++ for DOS/Windows. It introduced Object-Oriented Programming to geological stress analysis to handle the heavy computational load of evaluating ~500,000 tensors.
* **2025 (Modernization):** Ported to Python 3. Using **Numba** (Just-In-Time compilation), this version achieves the same execution speed as C++ on modern multi-core processors while offering superior visualization via **Matplotlib**.

## 🧠 The Method

ROMSA solves the inverse problem of finding the stress tensor that best explains a population of faults measured in the field. It uses a comprehensive grid-search approach rather than numerical inversion, allowing for the visualization of the entire solution space and its stability.

### 1. The Right Dihedra Principle
If slickenlines (striae) on a fault represent the direction of maximum resolved shear stress, the principal compressive stress ($\sigma_1$) must lie within a specific "pressure" quadrant (dihedra), and the principal extension stress ($\sigma_3$) must lie in the opposing "tension" quadrant. This forms the basis of the Angelier & Mechler (1977) method.

### 2. Lisle's Constraint (The Improvement)
Standard Right Dihedra methods treat $\sigma_1$ and $\sigma_3$ independently. Lisle (1987) introduced a kinematic constraint: a specific $\sigma_1$ orientation is only valid if there exists a corresponding $\sigma_3$ (perpendicular to it) that *also* satisfies the fault movement.

### 3. The Algorithm
1.  **Scan:** The program generates a grid of thousands of potential $\sigma_1$ orientations on a stereonet.
2.  **Check:** For every potential $\sigma_1$, it scans all orthogonal directions to find the best possible $\sigma_3$ that satisfies Lisle's constraint for the maximum number of faults.
3.  **Map:** The result is a probability map where bright areas represent orientations consistent with the majority of the field data.

---

## 🛠️ Prerequisites

Before installing ROMSA, you only need **Python** installed on your computer.

### 🪟 Windows
1.  Download the **"Windows installer (64-bit)"** from [python.org](https://www.python.org/downloads/).
    * *Crucial:* When installing, check the box at the bottom that says **"Add Python to PATH"**.

### 🍎 macOS
1.  Download the **"macOS 64-bit universal2 installer"** from [python.org](https://www.python.org/downloads/macos/).
2.  Run the installer.

### 🐧 Linux
If you are on Linux, you can install Python via your package manager:

    sudo apt update
    sudo apt install python3 python3-pip

---

## ⬇️ Installation

### 1. Download the Software
The easiest way to get ROMSA is to download it as a ZIP file:

1.  Scroll to the top of this GitHub page.
2.  Click the green **<> Code** button.
3.  Select **Download ZIP**.
4.  Extract (Unzip) the folder to your Desktop or Documents.

### 2. Open a Terminal
Navigate to the extracted folder using your terminal:

* **Windows 11:** Open the extracted folder. Right-click anywhere in the empty space and select **"Open in Terminal"**.
* **Windows 10:** Open the extracted folder. Hold the **Shift** key, Right-click in the empty space, and select **"Open PowerShell window here"**.
* **macOS:** Open the extracted folder in Finder. Right-click (or Ctrl-click) inside the folder and select **"New Terminal at Folder"**.

### 3. Install Dependencies
Once your terminal is open in the correct folder, run the following command to install the necessary libraries:

**🪟 Windows:**

    py -m pip install -r requirements.txt

**🍎 macOS / 🐧 Linux:**

    pip3 install -r requirements.txt

---

## 💻 Usage

Run the program by pointing it to your data file:

    python romsa.py examples/data.dat

### Automatic Outputs
ROMSA automatically generates two files in the same folder as your input data:

1.  **`filename_plot.png`**: A high-resolution (300 DPI), print-quality image of the stereonet and results panel. This allows for immediate inclusion in reports or publications.
2.  **`filename_tensors.csv`**: A spreadsheet containing the raw data for the most likely stress tensors ($\sigma_1$, $\sigma_2$, $\sigma_3$ trends and plunges), sorted by probability.

### Interactive Window
After calculation, an interactive plot window will open.
* **Explore:** You can use the toolbar to zoom and pan around the stereonet.
* **Save:** You can manually save specific views or different file formats (SVG, PDF) using the "Save" (floppy disk) icon.

### Advanced Options
You can control the resolution of the grid search using the `--res` flag:

| Flag | Description |
| :--- | :--- |
| `--res low` | **Fast Preview** (~31k points). Good for quick checks. |
| `--res medium` | **Standard** (Default, ~125k points). Good balance of speed and detail. |
| `--res high` | **High Precision** (~500k points). Creates the smoothest plots for publication. |

    python romsa.py examples/data.dat --res high

---

## 📄 Input File Format

The input `.dat` file is a plain text file compatible with the original 1994 software.

* **Header (Optional):** The first line can be the integer count of faults. If omitted, the program auto-detects it.
* **Data:** Each row represents one fault with 6 columns:

| Column | Parameter | Description |
| :--- | :--- | :--- |
| 1 | **Dip** | Fault plane dip (0-90). |
| 2 | **Dip Dir** | Fault plane dip direction (0-360). |
| 3 | **Plunge** | Striae plunge (0-90). |
| 4 | **Trend** | Striae trend/direction (0-360). |
| 5 | **Vertical** | Sense of slip: `1` (Normal), `-1` (Reverse), `0` (Strike-Slip). |
| 6 | **Horizontal** | Sense of slip: `1` (Dextral), `-1` (Sinistral), `0` (Dip-Slip). |

**Example `data.dat`:**

    11
    10 4   52 116  1  0
    54 290 53 272  1  0
    85 240 6  151 -1  0
    ...

## 📚 References

1.  **Ciscato, B. (1994).** *Principal Stress Orientations from Faults: a C++ program*. Structural Geology and Personal Computers, 325-342.
2.  **Lisle, R. J. (1987).** *Principal stress orientations from faults: an additional constraint*. Annales Tectonicae 1: 155-158.
3.  **Lisle, R. J. (1988).** *ROMSA: a basic program for paleostress analysis using fault-striation data*. Computers & Geosciences 14: 255-259.
4.  **Angelier, J. & Mechler, P. (1977).** *Sur une méthode graphique de recherche des contraintes principales...*. Bull. Soc. Geol. France 19: 1309-1318.
5.  **McKenzie, D. P. (1969).** *The relation between fault plane solutions and the directions of the principal stresses*. Bull. Seismolog. Soc. America 59: 591-601.

## 📄 License

This project is licensed under the MIT License.

## 💾 Gallery

If you have created a fantastic plot and want to share it, I'm happy to host it here.
