"""
meidem/grids/kostogryz2022.py
==============================
Limb darkening coefficient interpolator
Grid: Kostogryz et al. (2022), MPS-ATLAS

Tables (parquet):
  table5.parquet — nonlinear law (4 coefficients): a1, a2, a3, a4
  table6.parquet — power-2 law: c, alpha

Available passbands: TESS, Kepler, CHEOPS, PLATO

Reference:
  Kostogryz, N. V. et al. (2022)
  https://doi.org/10.1051/0004-6361/202140376
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# Path to the tables within the package
_TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tables', 'kostogryz2022')

# Map: law → (file, coefficient columns)
_LAW_MAP = {
    'nonlinear': ('table5.parquet', ['a1', 'a2', 'a3', 'a4']),
    'power2'   : ('table6.parquet', ['c', 'alpha']),
}

_VALID_PASSBANDS = ['TESS', 'Kepler', 'CHEOPS', 'PLATO']
_VALID_LAWS      = list(_LAW_MAP.keys())


class KostogryzGrid:
    """
    Tri-linear interpolator (Teff, logg, [Fe/H]) for LD coefficients
    from the MPS-ATLAS grid of Kostogryz et al. (2022).

    Parameters
    ----------
    passband : str   — 'TESS' | 'Kepler' | 'CHEOPS' | 'PLATO'
    law      : str   — 'power2' (default) | 'nonlinear'
    verbose  : bool
    """

    def __init__(self, passband='TESS', law='power2', verbose=False):
        self.passband = passband
        self.law      = law.lower()
        self.verbose  = verbose

        if self.passband not in _VALID_PASSBANDS:
            raise ValueError(
                f"passband='{self.passband}' is not supported for Kostogryz+2022.\n"
                f"Available options: {_VALID_PASSBANDS}"
            )
        if self.law not in _LAW_MAP:
            raise ValueError(
                f"law='{self.law}' is not supported for Kostogryz+2022.\n"
                f"Available options: {_VALID_LAWS}"
            )

        self._table_file, self._col_names = _LAW_MAP[self.law]
        self._filepath = os.path.join(_TABLES_DIR, self._table_file)
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

        # The Kostogryz table has a 'passband' column — filter by it
        if 'passband' in df.columns:
            df = df[df['passband'] == self.passband].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(
                    f"Passband '{self.passband}' not found in the table.\n"
                    f"Available: {_VALID_PASSBANDS}"
                )

        # Check required columns
        required = ['Teff', 'logg', 'M/H'] + self._col_names
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing columns in Kostogryz+2022 table: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Drop NaN in coefficient columns
        df = df.dropna(subset=self._col_names).reset_index(drop=True)

        self._df   = df
        self._Teff = df['Teff'].values.astype(float)
        self._logg = df['logg'].values.astype(float)
        self._feh  = df['M/H'].values.astype(float)
        self._coeffs = [df[c].values.astype(float) for c in self._col_names]

        if self.verbose:
            print('=' * 62)
            print(f'KOSTOGRYZ+2022  |  MPS-ATLAS  |  {self.passband}  |  {self.law}')
            print(f'Coefficients: {self._col_names}  |  Points: {len(self._Teff)}')
            print(f'Teff : [{self._Teff.min():.0f} – {self._Teff.max():.0f}] K')
            print(f'logg : [{self._logg.min():.1f} – {self._logg.max():.1f}]')
            print(f'[Fe/H]: [{self._feh.min():.2f} – {self._feh.max():.2f}]')
            print('=' * 62)

    # ── Interpolator ──────────────────────────────────────────────────────────
    def _build_interpolator(self):
        pts = np.column_stack((self._Teff, self._logg, self._feh))
        self._interps = [LinearNDInterpolator(pts, c) for c in self._coeffs]
        self.teff_min, self.teff_max = self._Teff.min(), self._Teff.max()
        self.logg_min, self.logg_max = self._logg.min(), self._logg.max()
        self.feh_min,  self.feh_max  = self._feh.min(),  self._feh.max()

    # ── Interpolation ─────────────────────────────────────────────────────────
    def get_coefficients(self, teff, logg, feh):
        """
        Interpolate LD coefficients for the given stellar parameters.

        Parameters
        ----------
        teff : float — Effective temperature (K)
        logg : float — Surface gravity log g
        feh  : float — Metallicity [Fe/H]

        Returns
        -------
        list[float] — interpolated coefficients
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
                "Parameters outside the Kostogryz+2022 grid limits:\n  " + "\n  ".join(errs)
            )

        result = [float(np.atleast_1d(f(teff, logg, feh))[0]) for f in self._interps]

        if any(np.isnan(v) for v in result):
            raise ValueError(
                f"Interpolation returned NaN for Teff={teff}, logg={logg}, [Fe/H]={feh}.\n"
                f"The point may lie in a region not covered by the grid."
            )
        return result

    # ── Static method called by __init__.py ───────────────────────────────────
    @staticmethod
    def interpolate(teff, logg, feh, passband='TESS', law='power2', verbose=False, **kwargs):
        """
        Static interface for calls from meidem.__init__.

        Returns
        -------
        tuple(list[float], dict) — (coefficients, metadata)
        """
        grid   = KostogryzGrid(passband=passband, law=law, verbose=verbose)
        coeffs = grid.get_coefficients(teff, logg, feh)
        meta   = {
            'col_names': grid._col_names,
            'teff_min' : grid.teff_min, 'teff_max': grid.teff_max,
            'logg_min' : grid.logg_min, 'logg_max': grid.logg_max,
            'feh_min'  : grid.feh_min,  'feh_max' : grid.feh_max,
        }
        return coeffs, meta