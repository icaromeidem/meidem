"""
MEIDEM — Multi-grid Exoplanet Interpolator for limb DarkEning Models
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
    # ... (A SUA DOCSTRING INTACTA AQUI) ...
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
    ...