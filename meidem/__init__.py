"""
MEIDEM — Multi-grid Epic Interpolator for stellar limb DarkEning Models
=====================================================================
Unified public API for interpolating limb darkening coefficients
from multiple published grids.

Supported grids
---------------
  'kostogryz2022' : Kostogryz+2022, MPS-ATLAS
                    Laws: 'nonlinear' (4coeff), 'power2'
                    Passbands: TESS, Kepler, CHEOPS, PLATO

  'claret2022'    : Claret & Southworth 2022, A&A 664, A128
                    Laws: 'power2'
                    Passbands: TESS, Kepler, Gaia_G, Gaia_BP, Gaia_RP,
                               SDSS_u, SDSS_g, SDSS_r, SDSS_i, SDSS_z,
                               uvby, Johnson_U/B/V/R/I, 2MASS_J/H/K

  'claret2017'    : Claret 2017, A&A 600, A30
                    Laws: 'quadratic', 'square-root', 'logarithmic',
                           '4coeff', 'linear', 'y'
                    Passbands: TESS
                    Models: ATLAS (mod='A'), PHOENIX (mod='P')

  'claret2011'    : Claret & Bloemen 2011, A&A 529, A75
                    Laws: 'quadratic', 'root-square', 'logarithmic',
                           '4coeff', 'linear', 'y'
                    Passbands: Kepler (Kp), CoRoT (C), Spitzer (S1, S2)
                    Models: ATLAS (mod='A'), PHOENIX (mod='P')

  'claret2025'    : Claret et al. 2025, A&A 699, A97
                    Laws: 'power2', '4coeff'
                    Passbands: JWST (F210M, F277W, F322W2, F444W, G235H,
                                     G235M, G395H, G395M, PRISM, SOSS1, SOSS2)
                    Models: ATLAS

    'magic2013'   : Magic et al. (2015), A&A 573, A90
                    Laws: 'linear', 'quadratic', 'square-root', '4coeff'
                    Passbands: Kepler, CoRoT,
                               Bessell_H/J/K,
                               Johnson_U/B/V/R/I/J/K,
                               SDSS_u/g/r/i/z,
                               Stromgren_u/v/b/y,
                               MK_J/K/L/Lp/M,
                               WFC3_grism
                    Model: 3D RHD STAGGER-grid (no xi)

Quick start
-----------
# ... existing code ...
Quick start
-----------
>>> import meidem
>>> result = meidem.get_ld_coefficients(
...     teff=5778, logg=4.44, feh=0.0,
...     passband='TESS',
...     grid='kostogryz2022',
...     law='power2',
... )
>>> print(result['coefficients'])

>>> # List available grids and laws
>>> meidem.available_grids()
>>> meidem.available_laws('claret2017')
>>> meidem.available_passbands('claret2022')
"""

import warnings  # top-level import

from ._version import __version__
from .grids.kostogryz2022 import KostogryzGrid
from .grids.claret2022    import Claret2022Grid
from .grids.claret2017    import Claret2017Grid
from .grids.claret2011    import Claret2011Grid
from .grids.claret2025    import Claret2025Grid
from .grids.magic2013     import Magic2013Grid


__all__ = [
    '__version__',
    'get_ld_coefficients',
    'available_grids',
    'available_laws',
    'available_passbands',
]

# Cache to store already-loaded interpolators for faster MCMC and Grid Searches
_GRID_CACHE = {}

# ── Grid registry ─────────────────────────────────────────────────────────────

_GRIDS = {
    'kostogryz2022': {
        'class'     : KostogryzGrid,
        'reference' : 'Kostogryz et al. (2022)',
        'model'     : 'MPS-ATLAS',
        'laws'      : ['nonlinear', 'power2'],
        'passbands' : ['TESS', 'Kepler', 'CHEOPS', 'PLATO'],
        'doi'       : '10.1051/0004-6361/202140376',
        'xi' : None,   

    },
    'claret2022': {
        'class'     : Claret2022Grid,
        'reference' : 'Claret & Southworth (2022), A&A 664, A128',
        'model'     : 'ATLAS',
        'laws'      : ['power2'],
        'passbands' : ['TESS', 'Kepler', 'Gaia_G', 'Gaia_BP', 'Gaia_RP',
                       'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z',
                       'Johnson_U', 'Johnson_B', 'Johnson_V', 'Johnson_R', 'Johnson_I',
                       '2MASS_J', '2MASS_H', '2MASS_K'],
        'doi'       : '10.1051/0004-6361/202244278',
        'xi' : [0, 1, 2, 4, 8],

    },
    'claret2017': {
        'class'     : Claret2017Grid,
        'reference' : 'Claret (2017), A&A 600, A30',
        'model'     : 'ATLAS/PHOENIX',
        'laws'      : ['quadratic', 'square-root', 'logarithmic', '4coeff', 'linear', 'y'],
        'passbands' : ['TESS'],
        'doi'       : '10.1051/0004-6361/201629705',
        'xi' : [0, 1, 2, 4, 8],
    },
    'claret2011': {
        'class'     : Claret2011Grid,
        'reference' : 'Claret & Bloemen (2011), A&A 529, A75',
        'model'     : 'ATLAS/PHOENIX',
        'laws'      : ['quadratic', 'root-square', 'logarithmic', '4coeff', 'linear', 'y'],
        'passbands' : ['Kp', 'C', 'S1', 'S2'],
        'doi'       : '10.1051/0004-6361/201116451',
        'xi' : [0, 1, 2, 4, 8],

    },
    'claret2025': {
    'class'     : Claret2025Grid,
    'reference' : 'Claret et al. (2025), A&A 699, A97',
    'model'     : 'ATLAS',
    'laws'      : ['power2', '4coeff'],
    'passbands' : ['F210M', 'F277W', 'F322W2', 'F444W',
                   'G235H', 'G235M', 'G395H', 'G395M', 'PRISM',
                   'SOSS1', 'SOSS2'],
    'doi'       : '10.1051/0004-6361/202554578',
    'xi'        : None,  
    },

    'magic2013': {
        'class'     : Magic2013Grid,
        'reference' : 'Magic et al. (2015), A&A 573, A90',
        'model'     : '3D RHD STAGGER-grid',
        'laws'      : ['linear', 'quadratic', 'square-root', '4coeff'],
        'passbands' : [
            'Kepler', 'CoRoT',
            'Bessell_H', 'Bessell_J', 'Bessell_K',
            'Johnson_U', 'Johnson_B', 'Johnson_V', 'Johnson_R', 'Johnson_I',
            'Johnson_J', 'Johnson_K',
            'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z',
            'Stromgren_u', 'Stromgren_v', 'Stromgren_b', 'Stromgren_y',
            'MK_J', 'MK_K', 'MK_L', 'MK_Lp', 'MK_M',
            'WFC3_grism',
        ],
        'doi'       : '10.1051/0004-6361/201423804',
        'xi'        : None,
    },
}

# ── Main API ──────────────────────────────────────────────────────────────────

_ATLAS_GRIDS = {'claret2022', 'claret2017', 'claret2011'}
_XI_DEFAULT = object() 

def get_ld_coefficients(
    teff,
    logg,
    feh,
    passband,
    grid    = 'kostogryz2022',
    law     = None,
    xi      = _XI_DEFAULT,   
    met     = 'L',
    mod     = 'A',
    verbose = False,
):
    
    """
    Interpolate limb darkening coefficients from published grids.

    Parameters
    ----------
    teff     : float
        Stellar effective temperature (K).
    logg     : float
        Surface gravity log g (cgs).
    feh      : float
        Metallicity [Fe/H] (dex).
    passband : str
        Photometric passband. Depends on the chosen grid.
        E.g.: 'TESS', 'Kepler', 'Gaia_G', 'Kp', 'C'
    grid     : str, optional
        LD coefficient grid. Default: 'kostogryz2022'.
        Options: 'kostogryz2022', 'claret2022', 'claret2017', 'claret2011'
    law      : str or None, optional
        Limb darkening law. If None, uses the grid default:
          - kostogryz2022 → 'power2'
          - claret2022    → 'power2'
          - claret2017    → 'quadratic'
          - claret2011    → 'quadratic'
    xi       : float, optional
        Microturbulence in km/s (ATLAS only). Default: 2.0.
        Available values: 0, 1, 2, 4, 8.
        Always set explicitly — if omitted, a UserWarning is emitted.
    met      : str, optional
        Fitting method (Claret 2017/2011 only).
        'L' = Least-Squares (default) | 'F' = Flux Conservation
    mod      : str, optional
        Atmospheric model (Claret 2017/2011 only).
        'A' = ATLAS (default) | 'P' = PHOENIX
    verbose  : bool, optional
        If True, prints grid info and coefficients. Default: False.

    Returns
    -------
    dict with keys:
        'coefficients' : list[float]  — interpolated LD coefficients
        'n_coeffs'     : int          — number of coefficients
        'law'          : str          — LD law used
        'passband'     : str          — passband used
        'grid'         : str          — grid used
        'reference'    : str          — bibliographic reference
        'doi'          : str          — paper DOI
        'teff_input'   : float        — input Teff
        'logg_input'   : float        — input logg
        'feh_input'    : float        — input [Fe/H]
        'xi'           : float|None   — microturbulence (None if PHOENIX or kostogryz2022)
        'met'          : str|None     — fitting method (None if not applicable)
        'mod'          : str|None     — atmospheric model (None if not applicable)

    Examples
    --------
    >>> import meidem

    >>> # Kostogryz+2022 — power-2 for TESS
    >>> r = meidem.get_ld_coefficients(5778, 4.44, 0.0, 'TESS', grid='kostogryz2022', law='power2')
    >>> r['coefficients']
    [0.412, 0.631]

    >>> # Claret & Southworth 2022 — power-2 for Kepler
    >>> r = meidem.get_ld_coefficients(5778, 4.44, 0.0, 'Kepler', grid='claret2022', xi=2.0)
    >>> r['coefficients']
    [0.398, 0.618]

    >>> # Claret 2017 — quadratic for TESS, ATLAS, Least-Squares
    >>> r = meidem.get_ld_coefficients(5778, 4.44, 0.0, 'TESS',
    ...     grid='claret2017', law='quadratic', mod='A', met='L', xi=2.0)
    >>> r['coefficients']
    [0.321, 0.287]

    >>> # Claret & Bloemen 2011 — quadratic for Kepler
    >>> r = meidem.get_ld_coefficients(5778, 4.44, 0.0, 'Kp',
    ...     grid='claret2011', law='quadratic', xi=2.0)
    >>> r['coefficients']
    [0.415, 0.295]

    Raises
    ------
    ValueError
        If the grid, law, or passband are not supported, or if the stellar
        parameters are outside the grid limits.
    """

    # ── 1. Normalise and validate grid ────────────────────────
    grid = grid.lower()
    if grid not in _GRIDS:
        raise ValueError(
            f"Grid '{grid}' is not supported.\n"
            f"Available options: {list(_GRIDS.keys())}\n"
            f"Use meidem.available_grids() for details."
        )

    # ── 2. Warn se xi não foi passado explicitamente ──────────
    xi_explicit = xi is not _XI_DEFAULT
    if not xi_explicit:
        xi = 2.0  # aplica o default real aqui
        if grid in _ATLAS_GRIDS and mod.upper() == 'A':
            warnings.warn(
                "Using default microturbulence xi=2.0 km/s. "
                "Available values: 0, 1, 2, 4, 8 km/s. "
                "Set xi explicitly if your star requires a different value "
                "(see meidem documentation for guidelines by stellar type).",
                UserWarning,
                stacklevel=2,
            )

    grid_info = _GRIDS[grid]

    # ── 3. Default law per grid ───────────────────────────────
    _default_law = {
        'kostogryz2022': 'power2',
        'claret2022'   : 'power2',
        'claret2017'   : 'quadratic',
        'claret2011'   : 'quadratic',
    }
    if law is None:
        law = _default_law[grid]

    # ── 4. Get from Cache or Instantiate Interpolator ─────────
    InterpolatorClass = grid_info['class']
    
    
    cache_key = (grid, passband, law, xi, met, mod)

    if cache_key not in _GRID_CACHE:
        if grid == 'kostogryz2022':
            obj = InterpolatorClass(passband=passband, law=law, verbose=verbose)
        elif grid == 'claret2022':
            obj = InterpolatorClass(passband=passband, xi=xi, verbose=verbose)
        elif grid == 'claret2017':
            obj = InterpolatorClass(law=law, xi=xi, met=met, mod=mod, verbose=verbose)
        elif grid == 'claret2011':
            obj = InterpolatorClass(law=law, passband=passband, xi=xi, met=met, mod=mod, verbose=verbose)
        elif grid == 'claret2025': 
            obj = InterpolatorClass(passband=passband, law=law, verbose=verbose)
        elif grid == 'magic2013': 
            obj = InterpolatorClass(passband=passband, law=law, verbose=verbose)
        _GRID_CACHE[cache_key] = obj
    else:
        pass

    cached_grid = _GRID_CACHE[cache_key]
    coeffs = cached_grid.get_coefficients(teff, logg, feh)

    _uses_atlas_xi = grid in _ATLAS_GRIDS and mod.upper() == 'A'
    _uses_claret   = grid in ('claret2017', 'claret2011')

    return {
        'coefficients': coeffs,
        'n_coeffs'    : len(coeffs),
        'law'         : law,
        'passband'    : passband,
        'grid'        : grid,
        'reference'   : grid_info['reference'],
        'doi'         : grid_info['doi'],
        'teff_input'  : teff,
        'logg_input'  : logg,
        'feh_input'   : feh,
        'xi'          : xi  if _uses_atlas_xi else None,
        'met'         : met if _uses_claret   else None,
        'mod'         : mod if _uses_claret else ('A' if grid == 'claret2022' else None),    
    }



def available_grids(verbose=True):
    """
     List all LD grids available in MEIDEM.

    Returns
    -------
    list[str] — names of available grids
    """
    if verbose:
        print("=" * 65)
        print("MEIDEM — Available Limb Darkening Grids")
        print("=" * 65)
        for name, info in _GRIDS.items():
            print(f"\n  [{name}]")
            print(f"    Reference : {info['reference']}")
            print(f"    Model     : {info['model']}")
            print(f"    Laws      : {info['laws']}")
            print(f"    Passbands : {info['passbands']}")
            xi_str = str(info['xi']) + " km/s" if info['xi'] else "not applicable"
            print(f"    xi (km/s) : {xi_str}")
            print(f"    DOI       : {info['doi']}")
        print("=" * 65)
    return list(_GRIDS.keys())


def available_laws(grid, verbose=True):
    """
    List the LD laws available for a specific grid.

    Parameters
    ----------
    grid : str — grid name

    Returns
    -------
    list[str]
    """
    grid = grid.lower()
    if grid not in _GRIDS:
        raise ValueError(f"Grid '{grid}' not found. Use available_grids().")
    laws = _GRIDS[grid]['laws']
    if verbose:
        print(f"Available laws for '{grid}': {laws}")
    return laws


def available_passbands(grid, verbose=True):
    """
    List the passbands available for a specific grid.

    Parameters
    ----------
    grid : str — grid name

    Returns
    -------
    list[str]
    """
    grid = grid.lower()
    if grid not in _GRIDS:
        raise ValueError(f"Grid '{grid}' not found. Use available_grids().")
    passbands = _GRIDS[grid]['passbands']
    if verbose:
        print(f"Available passbands for '{grid}': {passbands}")
    return passbands