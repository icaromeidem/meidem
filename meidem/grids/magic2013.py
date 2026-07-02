"""
meidem/grids/magic2013.py
==========================
Limb darkening coefficients from Magic et al. (2015), A&A 573, A90.

Based on 3D radiation-hydrodynamic (RHD) STAGGER-grid stellar models.
No microturbulence parameter (xi) — this is a 3D grid.

Reference
---------
Magic, Z., Claudi, R., Chiavassa, A., & Asplund, M. (2015)
A&A 573, A90
DOI: 10.1051/0004-6361/201423804

Supported passbands
-------------------
Passband key    System    Band
-----------     ------    ----
'Kepler'        kepler    (blank)
'CoRoT'         corot     (blank)
'Bessell_H'     bessell   h
'Bessell_J'     bessell   j
'Bessell_K'     bessell   k
'Johnson_U'     johnson   u
'Johnson_B'     johnson   b
'Johnson_V'     johnson   v
'Johnson_R'     johnson   r
'Johnson_I'     johnson   i
'Johnson_J'     johnson   j
'Johnson_K'     johnson   k
'SDSS_u'        sdss      u
'SDSS_g'        sdss      g
'SDSS_r'        sdss      r
'SDSS_i'        sdss      i
'SDSS_z'        sdss      z
'Stromgren_u'   stromgren u
'Stromgren_v'   stromgren v
'Stromgren_b'   stromgren b
'Stromgren_y'   stromgren y
'MK_J'          mk        j
'MK_K'          mk        k
'MK_L'          mk        l
'MK_Lp'         mk        lp
'MK_M'          mk        m
'WFC3_grism'    wfc3      grism

Supported laws
--------------
'linear'       : 1 coefficient  (u)
'quadratic'    : 2 coefficients (a, b)
'square-root'  : 2 coefficients (c, d)
'4coeff'       : 4 coefficients (a1, a2, a3, a4)
"""

import pathlib
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# ── Passband registry ─────────────────────────────────────────────────────────

_PASSBAND_MAP = {
    'Kepler'      : ('kepler',    ''),
    'CoRoT'       : ('corot',     ''),
    'Bessell_H'   : ('bessell',   'h'),
    'Bessell_J'   : ('bessell',   'j'),
    'Bessell_K'   : ('bessell',   'k'),
    'Johnson_U'   : ('johnson',   'u'),
    'Johnson_B'   : ('johnson',   'b'),
    'Johnson_V'   : ('johnson',   'v'),
    'Johnson_R'   : ('johnson',   'r'),
    'Johnson_I'   : ('johnson',   'i'),
    'Johnson_J'   : ('johnson',   'j'),
    'Johnson_K'   : ('johnson',   'k'),
    'SDSS_u'      : ('sdss',      'u'),
    'SDSS_g'      : ('sdss',      'g'),
    'SDSS_r'      : ('sdss',      'r'),
    'SDSS_i'      : ('sdss',      'i'),
    'SDSS_z'      : ('sdss',      'z'),
    'Stromgren_u' : ('stromgren', 'u'),
    'Stromgren_v' : ('stromgren', 'v'),
    'Stromgren_b' : ('stromgren', 'b'),
    'Stromgren_y' : ('stromgren', 'y'),
    'MK_J'        : ('mk',        'j'),
    'MK_K'        : ('mk',        'k'),
    'MK_L'        : ('mk',        'l'),
    'MK_Lp'       : ('mk',        'lp'),
    'MK_M'        : ('mk',        'm'),
    'WFC3_grism'  : ('wfc3',      'grism'),
}

# ── Law registry ──────────────────────────────────────────────────────────────

_LAW_COLS = {
    'linear'      : ['u'],
    'quadratic'   : ['a', 'b'],
    'square-root' : ['c', 'd'],
    '4coeff'      : ['a1', 'a2', 'a3', 'a4'],
}

_TABLE_FILE = pathlib.Path(__file__).parent.parent / 'tables' / 'magic2013' / 'table1.parquet'


class Magic2013Grid:
    """
    Interpolator for Magic et al. (2015) 3D RHD limb darkening grid.

    Parameters
    ----------
    passband : str
        Photometric passband. See module docstring for available keys.
    law : str
        Limb darkening law. One of: 'linear', 'quadratic', 'square-root', '4coeff'.
    verbose : bool, optional
        If True, prints grid info on initialisation. Default: False.

    Raises
    ------
    ValueError
        If passband or law is not supported.
    FileNotFoundError
        If the table file is not found.
    """

    def __init__(self, passband='Kepler', law='quadratic', verbose=False):
        if passband not in _PASSBAND_MAP:
            raise ValueError(
                f"Passband '{passband}' is not available in Magic2013Grid.\n"
                f"Available passbands: {list(_PASSBAND_MAP.keys())}"
            )
        if law not in _LAW_COLS:
            raise ValueError(
                f"Law '{law}' is not supported by Magic2013Grid.\n"
                f"Available laws: {list(_LAW_COLS.keys())}"
            )
        if not _TABLE_FILE.exists():
            raise FileNotFoundError(
                f"Magic2013 table not found: {_TABLE_FILE}\n"
                f"Convert the TSV first: python convert_magic2013.py\n"
                f"Or download from: https://doi.org/10.26093/cds/vizier.35730090"
            )

        self.passband = passband
        self.law      = law
        self._cols    = _LAW_COLS[law]
        self._interp  = None

        self._load(verbose)

    def _load(self, verbose):
        """Load the Parquet table and build the interpolator for the requested passband."""

        df = pd.read_parquet(_TABLE_FILE)

        df['System'] = df['System'].astype(str).str.strip().str.lower()
        df['Band'] = df['Band'].fillna('').astype(str).str.strip().str.lower()
        df['Band'] = df['Band'].replace('nan', '')
        
        # Filter for the requested passband
        sys_key, band_key = _PASSBAND_MAP[self.passband]
        mask  = (df['System'] == sys_key.lower()) & (df['Band'] == band_key.lower())
        df_pb = df[mask].dropna(subset=self._cols).copy()

        if len(df_pb) == 0:
            raise ValueError(
                f"No data found for passband '{self.passband}' "
                f"(system='{sys_key}', band='{band_key}') in Magic2013 table. "
                f"Available in table: {df['System'].unique()[:5]} / {df['Band'].unique()[:5]}"
            )

        if verbose:
            print(f"[Magic2013Grid] passband='{self.passband}'  law='{self.law}'")
            print(f"  Grid points : {len(df_pb)}")
            print(f"  Teff range  : {df_pb['Teff'].min():.0f} – {df_pb['Teff'].max():.0f} K")
            print(f"  logg range  : {df_pb['logg'].min():.2f} – {df_pb['logg'].max():.2f} dex")
            print(f"  [Fe/H] range: {df_pb['[Fe/H]'].min():.2f} – {df_pb['[Fe/H]'].max():.2f} dex")

        points = df_pb[['Teff', 'logg', '[Fe/H]']].values
        values = df_pb[self._cols].values

        self._interp     = LinearNDInterpolator(points, values)
        self._teff_range = (float(df_pb['Teff'].min()),   float(df_pb['Teff'].max()))
        self._logg_range = (float(df_pb['logg'].min()),   float(df_pb['logg'].max()))
        self._feh_range  = (float(df_pb['[Fe/H]'].min()), float(df_pb['[Fe/H]'].max()))

    def get_coefficients(self, teff, logg, feh):
        """
        Interpolate limb darkening coefficients.

        Parameters
        ----------
        teff : float — effective temperature (K)
        logg : float — surface gravity (dex)
        feh  : float — metallicity [Fe/H] (dex)

        Returns
        -------
        list[float] — interpolated coefficients

        Raises
        ------
        ValueError
            If the point is outside the grid or in a coverage gap.
        """
        if not (self._teff_range[0] <= teff <= self._teff_range[1]):
            raise ValueError(
                f"Teff={teff} K is outside the Magic2013 grid range "
                f"[{self._teff_range[0]:.0f}, {self._teff_range[1]:.0f}] K."
            )
        if not (self._logg_range[0] <= logg <= self._logg_range[1]):
            raise ValueError(
                f"logg={logg} is outside the Magic2013 grid range "
                f"[{self._logg_range[0]:.2f}, {self._logg_range[1]:.2f}] dex."
            )
        if not (self._feh_range[0] <= feh <= self._feh_range[1]):
            raise ValueError(
                f"[Fe/H]={feh} is outside the Magic2013 grid range "
                f"[{self._feh_range[0]:.2f}, {self._feh_range[1]:.2f}] dex."
            )

        result = self._interp([[teff, logg, feh]])[0]

        if np.any(np.isnan(result)):
            raise ValueError(
                f"Interpolation returned NaN for Teff={teff}, logg={logg}, [Fe/H]={feh}. "
                f"The point may be in a coverage gap of the Magic2013 3D RHD grid."
            )

        return result.tolist()