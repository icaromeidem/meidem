"""
meidem/grids/claret2022.py
===========================
Limb darkening coefficient interpolator
Grid: Claret & Southworth (2022), A&A 664, A128 — ATLAS plane-parallel

Tables (parquet):
  table1.parquet — Gaia (BP, G, RP), Kepler, TESS       → power-2 (g, h)
  table2.parquet — SDSS (u, g, r, i, z)                 → power-2 (g, h)
  table3.parquet — uvby, Johnson UBVRI, 2MASS JHK        → power-2 (g, h)

Single law: power-2
  I(mu)/I(1) = 1 - g*(1 - mu^h)

Reference:
  Claret, A. & Southworth, J. (2022)
  A&A 664, A128
  https://doi.org/10.1051/0004-6361/202244278
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

_TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tables', 'claret2022')

# Map: passband → (file, g_column, h_column)
_PASSBAND_MAP = {
    # table1
    'TESS'      : ('table1.parquet', 'gT',  'hT'),
    'Kepler'    : ('table1.parquet', 'gK',  'hK'),
    'Gaia_G'    : ('table1.parquet', 'gG',  'hG'),
    'Gaia_BP'   : ('table1.parquet', 'gBP', 'hBP'),
    'Gaia_RP'   : ('table1.parquet', 'gRP', 'hRP'),
    # table2
    'SDSS_u'    : ('table2.parquet', 'gu',  'hu'),
    'SDSS_g'    : ('table2.parquet', 'gg',  'hg'),
    'SDSS_r'    : ('table2.parquet', 'gr',  'hr'),
    'SDSS_i'    : ('table2.parquet', 'gi',  'hi'),
    'SDSS_z'    : ('table2.parquet', 'gz',  'hz'),
    # table3
    'Stromgren_u': ('table3.parquet', 'gu', 'hu'),
    'Stromgren_v': ('table3.parquet', 'gv', 'hv'),
    'Stromgren_b': ('table3.parquet', 'gb', 'hb'),
    'Stromgren_y': ('table3.parquet', 'gy', 'hy'),
    'Johnson_U'  : ('table3.parquet', 'gU', 'hU'),
    'Johnson_B'  : ('table3.parquet', 'gB', 'hB'),
    'Johnson_V'  : ('table3.parquet', 'gV', 'hV'),
    'Johnson_R'  : ('table3.parquet', 'gR', 'hR'),
    'Johnson_I'  : ('table3.parquet', 'gI', 'hI'),
    '2MASS_J'    : ('table3.parquet', 'gJ', 'hJ'),
    '2MASS_H'    : ('table3.parquet', 'gH', 'hH'),
    '2MASS_K'    : ('table3.parquet', 'gK', 'hK'),
}

_VALID_PASSBANDS = list(_PASSBAND_MAP.keys())
_VALID_LAWS      = ['power2']


class Claret2022Grid:
    """
    Tri-linear interpolator (Teff, logg, [Fe/H]) for power-2 coefficients
    from Claret & Southworth (2022).

    Parameters
    ----------
    passband : str   — see _VALID_PASSBANDS
    xi       : float — microturbulence in km/s (0, 1, 2, 4, 8); default 2.0
    verbose  : bool
    """

    def __init__(self, passband='TESS', xi=2.0, verbose=False):
        self.passband = passband
        self.xi       = xi
        self.verbose  = verbose
        self.law      = 'power2'   # only law available in this grid

        if self.passband not in _PASSBAND_MAP:
            raise ValueError(
                f"passband='{self.passband}' is not supported for Claret & Southworth 2022.\n"
                f"Available options: {_VALID_PASSBANDS}"
            )

        self._table_file, self._col_g, self._col_h = _PASSBAND_MAP[self.passband]
        self._filepath = os.path.join(_TABLES_DIR, self._table_file)
        self._col_names = [self._col_g, self._col_h]
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

        # Filter by xi (column Vel)
        if 'Vel' in df.columns:
            avail_xi = sorted(df['Vel'].unique())
            df = df[df['Vel'] == self.xi].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(
                    f"xi={self.xi} is not available.\n"
                    f"Available values: {avail_xi}"
                )

        # Check required columns
        required = ['Teff', 'logg', 'Z', self._col_g, self._col_h]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing columns in Claret & Southworth 2022 table: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Drop NaN in coefficient columns
        df = df.dropna(subset=[self._col_g, self._col_h]).reset_index(drop=True)

        self._df   = df
        self._Teff = df['Teff'].values.astype(float)
        self._logg = df['logg'].values.astype(float)
        self._feh  = df['Z'].values.astype(float)
        self._g    = df[self._col_g].values.astype(float)
        self._h    = df[self._col_h].values.astype(float)

        if self.verbose:
            print('=' * 62)
            print(f'CLARET & SOUTHWORTH (2022)  |  ATLAS  |  {self.passband}  |  power-2')
            print(f'xi={self.xi} km/s  |  Coefficients: [{self._col_g}, {self._col_h}]  |  Points: {len(self._Teff)}')
            print(f'Teff : [{self._Teff.min():.0f} – {self._Teff.max():.0f}] K')
            print(f'logg : [{self._logg.min():.1f} – {self._logg.max():.1f}]')
            print(f'[Fe/H]: [{self._feh.min():.2f} – {self._feh.max():.2f}]')
            print('=' * 62)

    # ── Interpolator ──────────────────────────────────────────────────────────
    def _build_interpolator(self):
        pts = np.column_stack((self._Teff, self._logg, self._feh))
        self._interp_g = LinearNDInterpolator(pts, self._g)
        self._interp_h = LinearNDInterpolator(pts, self._h)
        self.teff_min, self.teff_max = self._Teff.min(), self._Teff.max()
        self.logg_min, self.logg_max = self._logg.min(), self._logg.max()
        self.feh_min,  self.feh_max  = self._feh.min(),  self._feh.max()

    # ── Interpolation ─────────────────────────────────────────────────────────
    def get_coefficients(self, teff, logg, feh):
        """
        Interpolate power-2 coefficients (g, h).

        Parameters
        ----------
        teff : float — Effective temperature (K)
        logg : float — Surface gravity log g
        feh  : float — Metallicity [Fe/H]

        Returns
        -------
        list[float] — [g, h]
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
                "Parameters outside the Claret & Southworth 2022 grid limits:\n  " + "\n  ".join(errs)
            )

        g_val = float(np.atleast_1d(self._interp_g(teff, logg, feh))[0])
        h_val = float(np.atleast_1d(self._interp_h(teff, logg, feh))[0])

        if np.isnan(g_val) or np.isnan(h_val):
            raise ValueError(
                f"Interpolation returned NaN for Teff={teff}, logg={logg}, [Fe/H]={feh}.\n"
                f"The point may lie in a region not covered by the grid."
            )
        return [g_val, h_val]

    # ── Static method called by __init__.py ───────────────────────────────────
    @staticmethod
    def interpolate(teff, logg, feh, passband='TESS', xi=2.0, verbose=False, **kwargs):
        """
        Static interface for calls from meidem.__init__.

        Returns
        -------
        tuple(list[float], dict) — (coefficients, metadata)
        """
        grid   = Claret2022Grid(passband=passband, xi=xi, verbose=verbose)
        coeffs = grid.get_coefficients(teff, logg, feh)
        meta   = {
            'col_names': grid._col_names,
            'xi'       : xi,
            'teff_min' : grid.teff_min, 'teff_max': grid.teff_max,
            'logg_min' : grid.logg_min, 'logg_max': grid.logg_max,
            'feh_min'  : grid.feh_min,  'feh_max' : grid.feh_max,
        }
        return coeffs, meta