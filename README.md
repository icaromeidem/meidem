#
<p align="center">
  <img src="logo/meidem2.png" alt="MEIDEM logo" width="435"/>
</p>

# MEIDEM - Multi-grid Epic Interpolator for stellar limb DarkEning Models

[![PyPI version](https://badge.fury.io/py/meidem.svg)](https://badge.fury.io/py/meidem)
[![Python](https://img.shields.io/pypi/pyversions/meidem)](https://pypi.org/project/meidem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/icaromeidem/meidem/actions/workflows/tests.yml/badge.svg)](https://github.com/icaromeidem/meidem/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20189894.svg)](https://doi.org/10.5281/zenodo.20189894)

MEIDEM is a Python package that provides a unified interface for interpolating stellar **limb darkening (LD) coefficients** from multiple published grids. Instead of dealing with different table formats, column names, and interpolation strategies for each grid, you call a single function — `get_ld_coefficients()` — and MEIDEM handles the rest.

All grid tables are **bundled with the package**: no manual downloads, no VizieR visits, no path configuration.

---

## What is Stellar Limb Darkening?

When a planet transits its host star, the transit light curve is not only shaped by the planet's size and orbital geometry — it is also strongly affected by how the stellar brightness varies across the disk. Stars are not uniformly bright: they are brighter at the center and dimmer toward the edge (the "limb"). This effect, called **limb darkening**, must be correctly accounted for to accurately determine the planetary radius and other transit parameters.

Limb darkening is typically described by a **mathematical law** parametrized by one or more coefficients that depend on the stellar type (Teff, logg, metallicity) and the photometric bandpass. These coefficients are computed from model stellar atmospheres and published as tabulated grids. MEIDEM interpolates those grids for any given set of stellar parameters.

---

## Installation

```bash
pip install meidem
```

---

## Quick Start

```python
import meidem

# Get power-2 LD coefficients for a Sun-like star observed with TESS
result = meidem.get_ld_coefficients(
    teff=5778,        # effective temperature (K)
    logg=4.44,        # surface gravity (cgs)
    feh=0.0,          # metallicity [Fe/H] (dex)
    passband='TESS',
    grid='kostogryz2022',
    law='power2',
)

print(result['coefficients'])   # [c, alpha]
print(result['reference'])      # full bibliographic reference
print(result['doi'])            # DOI of the source paper
```

---

## Solar Reference Example (IAU 2015 Values)

The example below uses the IAU 2015 nominal solar parameters as a sanity check. You can use it to verify the installation and compare results across grids.

```python
import meidem

# IAU 2015 nominal solar parameters
# Teff = 5772 K  (Prša et al. 2016, AJ 152, 41)
# logg = 4.438   (log10 of GM_sun / R_sun^2 in cgs)
# [Fe/H] = 0.0   (by definition)
SUN = dict(teff=5772, logg=4.438, feh=0.0)

print("LD coefficients for the Sun (IAU 2015)\n")

# Kostogryz+2022 — power-2 for TESS (no xi — MPS-ATLAS grid)
r = meidem.get_ld_coefficients(**SUN, passband='TESS',
                                grid='kostogryz2022', law='power2')
print(f"Kostogryz+2022  power-2    TESS   : {r['coefficients']}")
# Expected: [c ≈ 0.65, alpha ≈ 0.60]

# Kostogryz+2022 — nonlinear (4-coeff) for TESS
r = meidem.get_ld_coefficients(**SUN, passband='TESS',
                                grid='kostogryz2022', law='nonlinear')
print(f"Kostogryz+2022  nonlinear  TESS   : {r['coefficients']}")

# Claret & Southworth 2022 — power-2 for TESS
# Sun is a FGK dwarf → xi=2.0 km/s (Valenti & Fischer 2005)
r = meidem.get_ld_coefficients(**SUN, passband='TESS', grid='claret2022',
                                xi=2.0)
print(f"Claret2022      power-2    TESS   : {r['coefficients']}")

# Claret 2017 — quadratic for TESS, ATLAS model, Least-Squares
# Sun is a FGK dwarf → xi=2.0 km/s (Valenti & Fischer 2005)
r = meidem.get_ld_coefficients(**SUN, passband='TESS',
                                grid='claret2017', law='quadratic',
                                mod='A', met='L', xi=2.0)
print(f"Claret2017      quadratic  TESS   : {r['coefficients']}")

# Claret & Bloemen 2011 — quadratic for Kepler
# Sun is a FGK dwarf → xi=2.0 km/s (Valenti & Fischer 2005)
r = meidem.get_ld_coefficients(**SUN, passband='Kp',
                                grid='claret2011', law='quadratic',
                                xi=2.0)
print(f"Claret2011      quadratic  Kepler : {r['coefficients']}")
```

### Loading your own star table

```python
import pandas as pd
import meidem

# Load a table with stellar parameters
df = pd.read_csv('my_stars.csv')
# Required columns: Teff, logg, FeH

results = []
for _, row in df.iterrows():
    try:
        r = meidem.get_ld_coefficients(
            teff=row['Teff'],
            logg=row['logg'],
            feh=row['FeH'],
            passband='TESS',
            grid='kostogryz2022',
            law='power2',
        )
        results.append({
            'star' : row['hostname'],
            'LD_c1': r['coefficients'][0],
            'LD_c2': r['coefficients'][1],
            'status': 'OK',
        })
    except ValueError:
        results.append({'star': row['hostname'], 'status': 'OUT_OF_GRID'})

df_ld = pd.DataFrame(results)
print(df_ld)
```

---

## Supported Grids

MEIDEM currently supports four published grids. Each covers a different combination of atmospheric models, LD laws, and photometric passbands.

---

### 1. `kostogryz2022` — Kostogryz et al. (2022), MPS-ATLAS

**Reference:** Kostogryz, N. V. et al. (2022), A&A — [doi:10.1051/0004-6361/202140376](https://doi.org/10.1051/0004-6361/202140376)

**Atmospheric model:** MPS-ATLAS — a modern 1D plane-parallel model developed at the Max Planck Institute for Solar System Research. It covers a wide range of metallicities and is one of the most up-to-date grids available for exoplanet transit modeling.

**Passbands:** TESS, Kepler, CHEOPS, PLATO

**Grid limits:** Teff ≈ 3500–10000 K · logg ≈ 2.0–5.0 · [Fe/H] ≈ −5.0 to +1.0

**Microturbulence (xi):** Not applicable — this grid does not depend on xi.

**Laws available:**

| Law key | Coefficients | Equation |
|---------|-------------|---------|
| `power2` | c, alpha | I(μ)/I(1) = 1 − c(1 − μ^α) |
| `nonlinear` | a1, a2, a3, a4 | I(μ)/I(1) = 1 − Σ aₙ(1 − μ^(n/2)) |

**When to use:** This is the **recommended grid for TESS and Kepler** as of 2022. Use `law='power2'` for most transit fitting applications — the power-2 law shows the lowest residuals and the smallest bias in recovered Rp/R* compared to the quadratic law for TESS photometry (Maxted 2023). Note that stars cooler than ~3500 K (cool M dwarfs) fall outside this grid.

```python
r = meidem.get_ld_coefficients(
    teff=5778, logg=4.44, feh=0.0,
    passband='TESS',
    grid='kostogryz2022',
    law='power2',       # or 'nonlinear'
)
```

---

### 2. `claret2022` — Claret & Southworth (2022), A&A 664, A128

**Reference:** Claret, A. & Southworth, J. (2022), A&A 664, A128 — [doi:10.1051/0004-6361/202244278](https://doi.org/10.1051/0004-6361/202244278)

**Atmospheric model:** ATLAS (plane-parallel, 1D).

**Passbands:** TESS, Kepler, Gaia (G, BP, RP), SDSS (u, g, r, i, z), Johnson (U, B, V, R, I), 2MASS (J, H, K), Strömgren (u, v, b, y)

**Grid limits:** Teff ≈ 3500–50000 K · logg ≈ 0.0–5.0 · [Fe/H] ≈ −5.0 to +1.0

**Laws available:** Power-2 only

**Microturbulence (xi):** Available at ξ = 0, 1, 2, 4, 8 km/s. **Always set `xi` explicitly** — see the [xi selection guide](#which-microturbulence-xi) below.

**When to use:** Excellent choice when you need **power-2 coefficients for a wide range of passbands beyond TESS and Kepler** — especially Gaia, SDSS, or Johnson bands. Also useful for comparison with `kostogryz2022` for TESS and Kepler.

```python
r = meidem.get_ld_coefficients(
    teff=5778, logg=4.44, feh=0.0,
    passband='Gaia_G',   # or 'TESS', 'Kepler', 'SDSS_r', 'Johnson_V', ...
    grid='claret2022',
    xi=2.0,              # set explicitly: 0, 1, 2, 4, or 8 km/s
)
```

---

### 3. `claret2017` — Claret (2017), A&A 600, A30

**Reference:** Claret, A. (2017), A&A 600, A30 — [doi:10.1051/0004-6361/201629705](https://doi.org/10.1051/0004-6361/201629705)

**Atmospheric models:** ATLAS and PHOENIX

**Passbands:** TESS only

**Grid limits (ATLAS):** Teff ≈ 3500–50000 K · logg ≈ 0.0–5.0 · [Fe/H] ≈ −5.0 to +1.0 · ξ = 0–8 km/s

**Grid limits (PHOENIX):** Teff ≈ 1500–12000 K · logg ≈ 2.5–6.0 · [Fe/H] = 0.0 only

**Microturbulence (xi):** Applies to ATLAS only (ignored for PHOENIX). **Always set `xi` explicitly when using `mod='A'`** — see the [xi selection guide](#which-microturbulence-xi) below.

**Laws available:**

| Law key | N coeff | Notes |
|---------|---------|-------|
| `linear` | 1 | Avoid — produces biases up to ~3% in Rp/R* (Espinoza & Jordán 2016) |
| `quadratic` | 2 | Most widely used; fast computation (Kopal 1950; Kreidberg 2015) |
| `square-root` | 2 | Better for hot stars |
| `logarithmic` | 2 | Better performance than quadratic (Espinoza & Jordán 2016) |
| `4coeff` | 4 | Most flexible; best physical description of intensity profile |
| `y` | 1 | Gravity darkening coefficient; ATLAS only |

**Fitting methods (`met`):**

| Key | Method | When to use |
|-----|--------|-------------|
| `'L'` | LSM — Least-Squares | Minimizes residuals to the intensity profile. Most common; default. |
| `'F'` | PCM/CFM — Flux Conservation | Ensures integrated flux is conserved. Preferred for generating synthetic light curves. |

**Atmospheric models (`mod`):**

| Key | Model | When to use |
|-----|-------|-------------|
| `'A'` | ATLAS | Covers all metallicities; default for most stars |
| `'P'` | PHOENIX | More realistic for cool stars (Teff < 5000 K); [Fe/H] = 0.0 only |

**When to use:** This grid is ideal when you want to **compare multiple LD laws for TESS** — for example, to assess how much the choice of law affects the recovered Rp/R*. The `4coeff` law provides the most accurate physical description; `quadratic` is most compatible with transit fitting codes like batman.

```python
# ATLAS — quadratic, Least-Squares (most common setup)
r = meidem.get_ld_coefficients(
    teff=5778, logg=4.44, feh=0.0,
    passband='TESS',
    grid='claret2017',
    law='quadratic',
    mod='A',    # 'A' = ATLAS | 'P' = PHOENIX
    met='L',    # 'L' = Least-Squares | 'F' = Flux Conservation
    xi=2.0,     # set explicitly: 0, 1, 2, 4, or 8 km/s (ATLAS only)
)

# PHOENIX — better for cool stars (Teff < 5000 K), [Fe/H] = 0.0 only
# xi is not used for PHOENIX
r = meidem.get_ld_coefficients(
    teff=3800, logg=4.8, feh=0.0,
    passband='TESS',
    grid='claret2017',
    law='quadratic',
    mod='P',
)
```

---

### 4. `claret2011` — Claret & Bloemen (2011), A&A 529, A75

**Reference:** Claret, A. & Bloemen, S. (2011), A&A 529, A75 — [doi:10.1051/0004-6361/201116451](https://doi.org/10.1051/0004-6361/201116451)

**Atmospheric models:** ATLAS and PHOENIX

**Passbands:** Kepler (`Kp`), CoRoT (`C`), Spitzer ch1 (`S1`), Spitzer ch2 (`S2`)

**Laws available:** quadratic, root-square, logarithmic, 4coeff, linear, y

**Microturbulence (xi):** Same as Claret 2017 — applies to ATLAS only. **Always set `xi` explicitly when using `mod='A'`.**

**Fitting methods (`met`) and models (`mod`):** Same as Claret 2017.

**When to use:** This is the **standard reference grid for Kepler and CoRoT** — one of the most cited LD grids in the exoplanet literature. If you are working with Kepler light curves, this is the grid to use for comparison with published results.

```python
r = meidem.get_ld_coefficients(
    teff=5778, logg=4.44, feh=0.0,
    passband='Kp',          # 'Kp' = Kepler | 'C' = CoRoT | 'S1'/'S2' = Spitzer
    grid='claret2011',
    law='quadratic',
    mod='A',
    met='L',
    xi=2.0,                 # set explicitly: 0, 1, 2, 4, or 8 km/s (ATLAS only)
)
```

---

## Choosing the Right Grid and Law

### Which grid?

| Situation | Recommended |
|-----------|------------|
| TESS photometry, modern analysis | `kostogryz2022` |
| Kepler photometry | `claret2011` or `kostogryz2022` |
| Gaia, SDSS, Johnson, 2MASS passbands | `claret2022` |
| Cool stars (Teff < 3500 K) with TESS | `claret2017` with `mod='P'` |
| Comparing grids (e.g. for a paper) | `kostogryz2022` + `claret2022` |

### Which law?

The power-2 law shows the lowest residuals and the smallest bias in recovered transit parameters for TESS and Kepler photometry. The quadratic law is the most widely used in the exoplanet literature because it enables fast and efficient light curve computation in popular codes such as batman (Kreidberg 2015). Laws such as logarithmic, square-root, and 3-parameter do a better job than quadratic and linear when deriving transit parameters (Espinoza & Jordán 2016).

| Law | Recommendation |
|-----|---------------|
| `power2` | **Best overall for TESS/Kepler.** 2 parameters, excellent accuracy (Maxted 2023). |
| `quadratic` | Most compatible with transit codes. Fast, 2 parameters, widely tabulated. |
| `nonlinear` / `4coeff` | Best physical description. Ideal when fixing coefficients rather than fitting them. |
| `logarithmic` / `square-root` | Better than quadratic, less widely supported by transit codes. |
| `linear` | Avoid — biases up to ~3% in Rp/R* (Espinoza & Jordán 2016). |

### Which microturbulence (xi)?

The `xi` parameter applies only to ATLAS-based grids (`claret2022`, `claret2017` with `mod='A'`, `claret2011` with `mod='A'`). It has no effect on MPS-ATLAS (`kostogryz2022`) or PHOENIX models. **Always set `xi` explicitly** — if omitted, MEIDEM defaults to 2.0 km/s and will emit a `UserWarning`.

Microturbulence correlates primarily with Teff and logg (Bruntt et al. 2010; Valenti & Fischer 2005). The table below gives recommended values for the discrete grid points available in the Claret grids:

| Stellar type | logg range | Recommended xi | Reference |
|--------------|-----------|---------------|-----------|
| FGK main-sequence dwarfs | 4.0 – 5.0 | 2.0 km/s | Valenti & Fischer (2005) |
| Subgiants / metal-poor stars | 3.5 – 4.0 | 1.0 km/s | Bruntt et al. (2010) |
| Giants | 2.0 – 3.5 | 4.0 km/s | Bruntt et al. (2010) |
| Supergiants | < 2.0 | 8.0 km/s | Gray (2008) |
| Cool M dwarfs | — | use PHOENIX (`mod='P'`) | — |

Available values in the grids: **0, 1, 2, 4, 8** km/s.

> **Note:** These are guidelines based on typical values in the literature. For precision work, determine xi from a spectroscopic analysis of your target star (e.g. by requiring no trend between Fe i line abundance and equivalent width).

### Grid coverage and known limitations

Each grid covers a finite region of stellar parameter space. Stars outside the grid raise a `ValueError` — MEIDEM **never extrapolates silently**.

| Grid | Teff range (K) | ATLAS | PHOENIX |
|------|---------------|-------|---------|
| kostogryz2022 | ~3500–10000 | ✓ | — |
| claret2022 | ~3500–50000 | ✓ | — |
| claret2017 | ~3500–50000 / ~1500–12000 | ✓ | ✓ ([Fe/H]=0 only) |
| claret2011 | ~3500–50000 / ~1500–12000 | ✓ | ✓ ([Fe/H]=0 only) |

**Note for cool M dwarfs (Teff < 5000 K):** For stars cooler than about 5000 K, significant deviations between measured and theoretical LDCs (Δu₁, Δu₂ ≈ 0.2) have been observed in TESS light curves (Patel & Espinoza 2022). Treat results with extra caution for very cool stars.

---

## API Reference

### `meidem.get_ld_coefficients()`

```python
meidem.get_ld_coefficients(
    teff,                       # float — effective temperature (K)
    logg,                       # float — surface gravity log g (cgs)
    feh,                        # float — metallicity [Fe/H] (dex)
    passband,                   # str   — photometric passband
    grid    = 'kostogryz2022',  # str   — grid key
    law     = None,             # str   — LD law (None = grid default)
    xi      = 2.0,              # float — microturbulence km/s (ATLAS only; set explicitly)
    met     = 'L',              # str   — fitting method: 'L' or 'F'
    mod     = 'A',              # str   — atmospheric model: 'A' or 'P'
    verbose = False,            # bool  — print grid info
)
```

> **Note on `xi`:** If you use an ATLAS-based grid and do not set `xi` explicitly, MEIDEM defaults to 2.0 km/s and emits a `UserWarning`. Always pass `xi` explicitly to suppress the warning and ensure physically correct results. Available values: 0, 1, 2, 4, 8 km/s. See the [xi selection guide](#which-microturbulence-xi) for recommendations by stellar type.

**Returns:**

```python
{
    'coefficients': [c, alpha],       # interpolated LD coefficients
    'n_coeffs'    : 2,                # number of coefficients
    'law'         : 'power2',         # LD law used
    'passband'    : 'TESS',           # passband used
    'grid'        : 'kostogryz2022',  # grid used
    'reference'   : 'Kostogryz et al. (2022)',
    'doi'         : '10.1051/0004-6361/202140376',
    'teff_input'  : 5778.0,
    'logg_input'  : 4.44,
    'feh_input'   : 0.0,
    'xi'          : None,               # float if Claret ATLAS, else None
    'met'         : None,               # str if Claret 2017/2011, else None
    'mod'         : None,               # str if Claret 2017/2011, else None
}
```

**Raises `ValueError`** if parameters are outside grid limits.

### Discovery functions

```python
meidem.available_grids()                  # all grids with details
meidem.available_laws('claret2017')       # laws for a grid
meidem.available_passbands('claret2022')  # passbands for a grid
```

---

## Citation

If you use MEIDEM in your research, please cite the package and the grid(s) you used:

```bibtex
@software{meidem,
  author  = {Meidem, Icaro and Emilio, Marcelo},
  title   = {{MEIDEM}: Multi-grid Epic Interpolator for limb DarkEning Models},
  year    = {2026},
  version = {1.0.8},
  doi     = {10.5281/zenodo.20189894},
  url     = {https://github.com/icaromeidem/meidem},
}
```

**Grid references:**

- Kostogryz et al. (2022), A&A — [doi:10.1051/0004-6361/202140376](https://doi.org/10.1051/0004-6361/202140376)
- Claret & Southworth (2022), A&A 664, A128 — [doi:10.1051/0004-6361/202244278](https://doi.org/10.1051/0004-6361/202244278)
- Claret (2017), A&A 600, A30 — [doi:10.1051/0004-6361/201629705](https://doi.org/10.1051/0004-6361/201629705)
- Claret & Bloemen (2011), A&A 529, A75 — [doi:10.1051/0004-6361/201116451](https://doi.org/10.1051/0004-6361/201116451)

**Key references used in this documentation:**

- Bruntt, H. et al. (2010), MNRAS 405, 1907 — [doi:10.1111/j.1365-2966.2010.16545.x](https://doi.org/10.1111/j.1365-2966.2010.16545.x)
- Espinoza & Jordán (2015), MNRAS 450, 1879
- Espinoza & Jordán (2016), MNRAS 457, 3573
- Gray, D. F. (2008), *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge University Press
- Maxted (2023), MNRAS 519, 3723
- Patel & Espinoza (2022), AJ 163, 228
- Valenti, J. A. & Fischer, D. A. (2005), ApJS 159, 141 — [doi:10.1086/430500](https://doi.org/10.1086/430500)

---

## License

MIT © Icaro Meidem