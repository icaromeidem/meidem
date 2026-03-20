# MEIDEM 🌟

**Multi-grid Exoplanet Interpolator for limb DarkEning Models**

[![PyPI version](https://badge.fury.io/py/meidem.svg)](https://badge.fury.io/py/meidem)
[![Python](https://img.shields.io/pypi/pyversions/meidem)](https://pypi.org/project/meidem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MEIDEM is a Python package for interpolating stellar limb darkening (LD) coefficients from multiple published grids. It provides a single, unified API regardless of which grid or photometric passband you need — just call `get_ld_coefficients()` and you're done.

---

## Supported Grids

| Grid key | Reference | Model | Laws | Passbands |
|----------|-----------|-------|------|-----------|
| `kostogryz2022` | Kostogryz et al. (2022) | MPS-ATLAS | nonlinear, power-2 | TESS, Kepler, CHEOPS, PLATO |
| `claret2022` | Claret & Southworth (2022), A&A 664, A128 | ATLAS | power-2 | TESS, Kepler, Gaia, SDSS, Johnson, 2MASS |
| `claret2017` | Claret (2017), A&A 600, A30 | ATLAS / PHOENIX | quadratic, square-root, logarithmic, 4coeff, linear, y | TESS |
| `claret2011` | Claret & Bloemen (2011), A&A 529, A75 | ATLAS / PHOENIX | quadratic, root-square, logarithmic, 4coeff, linear, y | Kepler, CoRoT, Spitzer |

---

## Installation

```bash
pip install meidem
```

All grid tables are bundled with the package — no manual downloads required.

---

## Quick Start

```python
import meidem

# Kostogryz+2022 — power-2 for TESS
result = meidem.get_ld_coefficients(
    teff=5778, logg=4.44, feh=0.0,
    passband='TESS',
    grid='kostogryz2022',
    law='power2',
)
print(result['coefficients'])   # [c, alpha]
print(result['reference'])      # 'Kostogryz et al. (2022)'

# Claret & Southworth 2022 — power-2 for Kepler
result = meidem.get_ld_coefficients(
    teff=5778, logg=4.44, feh=0.0,
    passband='Kepler',
    grid='claret2022',
)
print(result['coefficients'])   # [g, h]

# Claret 2017 — quadratic for TESS, ATLAS model
result = meidem.get_ld_coefficients(
    teff=5778, logg=4.44, feh=0.0,
    passband='TESS',
    grid='claret2017',
    law='quadratic',
    mod='A',   # 'A' = ATLAS | 'P' = PHOENIX
    met='L',   # 'L' = Least-Squares | 'F' = Flux Conservation
)
print(result['coefficients'])   # [a, b]

# Claret & Bloemen 2011 — quadratic for Kepler
result = meidem.get_ld_coefficients(
    teff=5778, logg=4.44, feh=0.0,
    passband='Kp',
    grid='claret2011',
    law='quadratic',
)
print(result['coefficients'])   # [a, b]
```

---

## Discovery Functions

```python
# List all available grids
meidem.available_grids()

# List available LD laws for a specific grid
meidem.available_laws('claret2017')

# List available passbands for a specific grid
meidem.available_passbands('claret2022')
```

---

## Output Format

Every call to `get_ld_coefficients()` returns a standardized dictionary:

```python
{
    'coefficients': [0.412, 0.631],   # interpolated LD coefficients
    'n_coeffs'    : 2,                # number of coefficients
    'law'         : 'power2',         # LD law used
    'passband'    : 'TESS',           # photometric passband
    'grid'        : 'kostogryz2022',  # grid used
    'reference'   : 'Kostogryz et al. (2022)',
    'doi'         : '10.1051/0004-6361/202140376',
    'teff_input'  : 5778.0,
    'logg_input'  : 4.44,
    'feh_input'   : 0.0,
    'xi'          : None,
    'met'         : None,
    'mod'         : None,
}
```

---

## Citation

If you use MEIDEM in your research, please cite the package and the relevant grid(s):

```bibtex
@software{meidem,
  author  = {Meidem, Icaro},
  title   = {{MEIDEM}: Multi-grid Exoplanet Interpolator for limb DarkEning Models},
  year    = {2025},
  url     = {https://github.com/icaromeidem/meidem},
}
```

And the appropriate grid references:

- **Kostogryz+2022**: Kostogryz et al. 2022, A&A — doi:[10.1051/0004-6361/202140376](https://doi.org/10.1051/0004-6361/202140376)
- **Claret & Southworth 2022**: A&A 664, A128 — doi:[10.1051/0004-6361/202244278](https://doi.org/10.1051/0004-6361/202244278)
- **Claret 2017**: A&A 600, A30 — doi:[10.1051/0004-6361/201629705](https://doi.org/10.1051/0004-6361/201629705)
- **Claret & Bloemen 2011**: A&A 529, A75 — doi:[10.1051/0004-6361/201116451](https://doi.org/10.1051/0004-6361/201116451)

---

## License

MIT © Icaro Meidem
