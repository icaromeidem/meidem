"""
meidem/grids/claret2011.py
===========================
Limb darkening coefficient interpolator
Grid: Claret & Bloemen (2011), A&A 529, A75

Tables (parquet):
  table-af.parquet  → quadratic (a, b), root-square (c, d), logarithmic (e, f)
  tableeq5.parquet  → 4coeff (a1, a2, a3, a4)
  tableu.parquet    → linear (u)
  tabley.parquet    → y / exponential (y)

Passbands: Kepler (Kp), CoRoT (C), Spitzer ch1 (S1), Spitzer ch2 (S2)

Method: "L" = Least-Squares  |  "F" = Flux Conservation
Model:  "A" = ATLAS           |  "P" = PHOENIX

Reference:
  Claret, A. & Bloemen, S. (2011)
  A&A 529, A75
  https://doi.org/10.1051/0004-6361/201116451
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

_TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tables', 'claret2011')

_LAW_TABLE_MAP = {
    'quadratic'  : 'table-af.parquet',
    'root-square': 'table-af.parquet',
    'logarithmic': 'table-af.parquet',
    '4coeff'     : 'tableeq5.parquet',
    'linear'     : 'tableu.parquet',
    'y'          : 'tabley.parquet',
}

_LAW_COLUMNS = {
    'quadratic'  : ['a', 'b'],
    'root-square': ['c', 'd'],
    'logarithmic': ['e', 'f'],
    '4coeff'     : ['a1', 'a2', 'a3', 'a4'],
    'linear'     : ['u'],
    'y'          : ['y'],
}

_PASSBAND_MAP = {
    'Kp': 'Kp',
    'C' : 'C',
    'S1': 'S1',
    'S2': 'S2',
}

_VALID_LAWS      = list(_LAW_TABLE_MAP.keys())
_VALID_PASSBANDS = list(_PASSBAND_MAP.keys())
_VALID_MET       = {'L', 'F'}
_VALID_MOD       = {'A', 'P'}


class Claret2011Grid:
    """
    Tri-linear interpolator (Teff, logg, Z) for LD coefficients
    from Claret & Bloemen (2011).

    Parameters
    ----------
    law      : str   — 'quadratic' (default) | 'root-square' | 'logarithmic' |
                       '4coeff' | 'linear' | 'y'
    passband : str   — 'Kp' (Kepler) | 'C' (CoRoT) | 'S1' | 'S2' (Spitzer)
    xi       : float — microturbulence in km/s (default 2.0)
    met      : str   — 'L' (Least-Squares, default) | 'F' (Flux Conservation)
    mod      : str   — 'A' (ATLAS, default) | 'P' (PHOENIX)
    verbose  : bool
    """

    def __init__(self, law='quadratic', passband='Kp', xi=2.0,
                 met='L', mod='A', verbose=False):
        self.law      = law.lower()
        self.passband = passband
        self.xi       = xi
        self.met      = met.upper()
        self.mod      = mod.upper()
        self.verbose  = verbose

        if self.law not in _LAW_TABLE_MAP:
            raise ValueError(
                f"law='{self.law}' is not supported for Claret & Bloemen 2011.\n"
                f"Available options: {_VALID_LAWS}"
            )
        if self.passband not in _PASSBAND_MAP:
            raise ValueError(
                f"passband='{self.passband}' is not supported for Claret & Bloemen 2011.\n"
                f"Available options: {_VALID_PASSBANDS}"
            )
        if self.met not in _VALID_MET:
            raise ValueError(f"met='{self.met}' is invalid. Use 'L' (Least-Squares) or 'F' (Flux Conservation).")
        if self.mod not in _VALID_MOD:
            raise ValueError(f"mod='{self.mod}' is invalid. Use 'A' (ATLAS) or 'P' (PHOENIX).")

        self._filepath  = os.path.join(_TABLES_DIR, _LAW_TABLE_MAP[self.law])
        self._col_names = list(_LAW_COLUMNS[self.law])
        self._load_table()
        self._build_interpolator()

    # ── Table loading ─────────────────────────────────────────────────────────
    def _load_table(self):
        if not os.path.exists(self._filepath):
            raise FileNotFoundError(
                f"Table not found: {self._filepath}\n"
                f"Please verify that the tables were correctly installed with the package."
            )

        df = pd.read_parquet(self._filepath)

        # Filter by xi
        if 'xi' in df.columns:
            df = df[df['xi'] == self.xi]

        # Filter by passband
        filt = _PASSBAND_MAP[self.passband]
        if 'Filt' in df.columns:
            df = df[df['Filt'] == filt]

        # Filter by method (Met)
        if 'Met' in df.columns:
            df = df[df['Met'] == self.met]

        # Filter by model (Mod)
        if 'Mod' in df.columns:
            df = df[df['Mod'] == self.mod]

        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No data found for passband='{self.passband}', "
                f"xi={self.xi}, met='{self.met}', mod='{self.mod}'.\n"
                f"Please check the parameters and the available table."
            )

        # Check required columns
        missing = [c for c in self._col_names if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing columns in Claret & Bloemen 2011 table: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Drop NaN in coefficient columns
        df = df.dropna(subset=self._col_names).reset_index(drop=True)

        self._df     = df
        self._Teff   = df['Teff'].values.astype(float)
        self._logg   = df['logg'].values.astype(float)
        self._feh    = df['Z'].values.astype(float)
        self._coeffs = [df[c].values.astype(float) for c in self._col_names]

        if self.verbose:
            mod_lbl = 'ATLAS' if self.mod == 'A' else 'PHOENIX'
            met_lbl = 'Least-Squares' if self.met == 'L' else 'Flux Conservation'
            print('=' * 62)
            print(f'CLARET & BLOEMEN (2011)  |  {self.passband}  |  {self.law}')
            print(f'Model: {mod_lbl}  |  Method: {met_lbl}  |  xi={self.xi} km/s')
            print(f'Coefficients: {self._col_names}  |  Points: {len(self._Teff)}')
            print(f'Teff : [{self._Teff.min():.0f} – {self._Teff.max():.0f}] K')
            print(f'logg : [{self._logg.min():.1f} – {self._logg.max():.1f}]')
            print(f'[Fe/H]: [{self._feh.min():.2f} – {self._feh.max():.2f}]')
            print('=' * 62)

    # ── Interpolator ──────────────────────────────────────────────────────────
    def _build_interpolator(self):
        # If Z is constant (e.g. PHOENIX Z=0), use 2D interpolator (Teff, logg)
        self._feh_fixed = None
        if len(np.unique(self._feh)) == 1:
            self._feh_fixed = self._feh[0]
            pts = np.column_stack((self._Teff, self._logg))
        else:
            pts = np.column_stack((self._Teff, self._logg, self._feh))

        self._interps = [LinearNDInterpolator(pts, c) for c in self._coeffs]
        self.teff_min, self.teff_max = self._Teff.min(), self._Teff.max()
        self.logg_min, self.logg_max = self._logg.min(), self._logg.max()
        self.feh_min,  self.feh_max  = self._feh.min(),  self._feh.max()

    # ── Interpolation ─────────────────────────────────────────────────────────
    def get_coefficients(self, teff, logg, feh):
        """
        Interpolate LD coefficients.

        Parameters
        ----------
        teff : float — Effective temperature (K)
        logg : float — Surface gravity log g
        feh  : float — Metallicity [Fe/H]

        Returns
        -------
        list[float]
        """
        errs = []
        if not (self.teff_min <= teff <= self.teff_max):
            errs.append(f"Teff={teff:.0f} outside [{self.teff_min:.0f}, {self.teff_max:.0f}] K")
        if not (self.logg_min <= logg <= self.logg_max):
            errs.append(f"logg={logg:.2f} outside [{self.logg_min:.1f}, {self.logg_max:.1f}]")
        if not (self.feh_min <= feh <= self.feh_max):
            errs.append(f"[Fe/H]={feh:.2f} outside [{self.feh_min:.2f}, {self.feh_max:.2f}]")
        if errs:
            raise ValueError(
                "Parameters outside the Claret & Bloemen 2011 grid limits:\n  " + "\n  ".join(errs)
            )

        # PHOENIX grid: fixed Z → 2D interpolation (Teff, logg)
        if self._feh_fixed is not None:
            result = [float(np.atleast_1d(f(teff, logg))[0]) for f in self._interps]
        else:
            result = [float(np.atleast_1d(f(teff, logg, feh))[0]) for f in self._interps]

        if any(np.isnan(v) for v in result):
            raise ValueError(
                f"Interpolation returned NaN for Teff={teff}, logg={logg}, [Fe/H]={feh}.\n"
                f"The point may lie in a region not covered by the grid."
            )
        return result

    # ── Static method called by __init__.py ───────────────────────────────────
    @staticmethod
    def interpolate(teff, logg, feh, passband='Kp', law='quadratic',
                    xi=2.0, met='L', mod='A', verbose=False, **kwargs):
        """
        Static interface for calls from meidem.__init__.

        Returns
        -------
        tuple(list[float], dict) — (coefficients, metadata)
        """
        grid   = Claret2011Grid(law=law, passband=passband, xi=xi,
                                met=met, mod=mod, verbose=verbose)
        coeffs = grid.get_coefficients(teff, logg, feh)
        meta   = {
            'col_names': grid._col_names,
            'xi'       : xi,
            'met'      : met,
            'mod'      : mod,
            'teff_min' : grid.teff_min, 'teff_max': grid.teff_max,
            'logg_min' : grid.logg_min, 'logg_max': grid.logg_max,
            'feh_min'  : grid.feh_min,  'feh_max' : grid.feh_max,
        }
        return coeffs, meta