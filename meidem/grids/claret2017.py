"""
meidem/grids/claret2017.py
===========================
Interpolador de coeficientes de limb darkening
Grade: Claret (2017), A&A 600, A30 — banda TESS

Tabelas PHOENIX (sem xi, Z=0 apenas):
  tableab.parquet  → quadratic    (aLSM/bLSM, aPCM/bPCM)
  tablecd.parquet  → square-root  (cLSM/dLSM, cPCM/dPCM)
  tableef.parquet  → logarithmic  (eLSM/fLSM, ePCM/fPCM)
  tablea.parquet   → 4coeff       (a1LSM..a4LSM)
  tableu.parquet   → linear       (uLSM, uPCM)

Tabelas ATLAS (com xi, todas metalicidades):
  table25.parquet  → quadratic
  table26.parquet  → square-root
  table27.parquet  → logarithmic
  table28.parquet  → 4coeff
  table24.parquet  → linear       (uLSM, uCFM)
  table29.parquet  → y/GDC        (y) — usa logTeff

Método: "L" = LSM (Least-Squares)  |  "F" = PCM/CFM (Flux Conservation)
Modelo: "A" = ATLAS                |  "P" = PHOENIX

Referência:
  Claret, A. (2017)
  A&A 600, A30
  https://doi.org/10.1051/0004-6361/201629705
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

_TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tables', 'claret2017')

# (arquivo_phoenix, arquivo_atlas)
_LAW_TABLE_MAP = {
    'quadratic'  : ('tableab.parquet',  'table25.parquet'),
    'square-root': ('tablecd.parquet',  'table26.parquet'),
    'logarithmic': ('tableef.parquet',  'table27.parquet'),
    '4coeff'     : ('tablea.parquet',   'table28.parquet'),
    'linear'     : ('tableu.parquet',   'table24.parquet'),
    'y'          : (None,               'table29.parquet'),
}

# Colunas LSM e PCM/CFM por lei
_LAW_COLUMNS = {
    'quadratic'  : {'L': ['aLSM', 'bLSM'],                       'F': ['aPCM', 'bPCM']},
    'square-root': {'L': ['cLSM', 'dLSM'],                       'F': ['cPCM', 'dPCM']},
    'logarithmic': {'L': ['eLSM', 'fLSM'],                       'F': ['ePCM', 'fPCM']},
    '4coeff'     : {'L': ['a1LSM', 'a2LSM', 'a3LSM', 'a4LSM'],   'F': ['a1LSM', 'a2LSM', 'a3LSM', 'a4LSM']},
    'linear'     : {'L': ['uLSM'],                                'F': ['uPCM']},
    'y'          : {'L': ['y'],                                   'F': ['y']},
}

_VALID_LAWS      = list(_LAW_TABLE_MAP.keys())
_VALID_PASSBANDS = ['TESS']
_VALID_MET       = {'L', 'F'}
_VALID_MOD       = {'A', 'P'}


class Claret2017Grid:
    """
    Interpolador tri-linear (Teff, logg, Z) para LD Claret 2017 — TESS.

    Parâmetros
    ----------
    law     : str   — 'quadratic' (default) | 'square-root' | 'logarithmic' |
                      '4coeff' | 'linear' | 'y'
    xi      : float — microturbulência km/s, só ATLAS (0, 1, 2, 4, 8); default 2.0
    met     : str   — 'L' (Least-Squares, default) | 'F' (Flux Conservation)
    mod     : str   — 'A' (ATLAS, default) | 'P' (PHOENIX)
    verbose : bool
    """

    def __init__(self, law='quadratic', xi=2.0, met='L', mod='A', verbose=False):
        self.law     = law.lower()
        self.xi      = xi
        self.met     = met.upper()
        self.mod     = mod.upper()
        self.verbose = verbose
        self.passband = 'TESS'  # única passband desta grade

        if self.law not in _LAW_TABLE_MAP:
            raise ValueError(
                f"law='{self.law}' não suportada para Claret 2017.\n"
                f"Opções: {_VALID_LAWS}"
            )
        if self.met not in _VALID_MET:
            raise ValueError(f"met='{self.met}' inválido. Use 'L' (Least-Squares) ou 'F' (Flux Conservation).")
        if self.mod not in _VALID_MOD:
            raise ValueError(f"mod='{self.mod}' inválido. Use 'A' (ATLAS) ou 'P' (PHOENIX).")
        if self.mod == 'P' and self.law == 'y':
            raise ValueError("Lei 'y' (GDC) só disponível para mod='A' (ATLAS).")

        phoenix_file, atlas_file = _LAW_TABLE_MAP[self.law]
        if self.mod == 'P' and phoenix_file is None:
            raise ValueError(f"Lei '{self.law}' não disponível para PHOENIX.")

        fname = atlas_file if self.mod == 'A' else phoenix_file
        self._filepath  = os.path.join(_TABLES_DIR, fname)
        self._col_names = list(_LAW_COLUMNS[self.law][self.met])
        self._load_table()
        self._build_interpolator()

    # ── Carregamento ──────────────────────────────────────────────────────────
    def _load_table(self):
        if not os.path.exists(self._filepath):
            raise FileNotFoundError(
                f"Tabela não encontrada: {self._filepath}\n"
                f"Verifique se as tabelas foram instaladas corretamente com o pacote."
            )

        df = pd.read_parquet(self._filepath)

        # Filtro xi (apenas ATLAS)
        if 'xi' in df.columns:
            avail_xi = sorted(df['xi'].unique())
            df = df[df['xi'] == self.xi].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(
                    f"xi={self.xi} não disponível.\n"
                    f"Valores disponíveis: {avail_xi}"
                )

        # table24 (linear ATLAS) usa 'uCFM' em vez de 'uPCM'
        if self.law == 'linear' and self.mod == 'A' and self.met == 'F':
            if 'uCFM' in df.columns:
                self._col_names = ['uCFM']

        # table29 (y/GDC) usa logTeff
        if 'logTeff' in df.columns and 'Teff' not in df.columns:
            df = df.copy()
            df['Teff'] = 10.0 ** df['logTeff']

        # Verifica colunas
        missing = [c for c in self._col_names if c not in df.columns]
        if missing:
            raise KeyError(
                f"Colunas ausentes na tabela Claret 2017: {missing}\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

        # Remove NaN nos coeficientes
        df = df.dropna(subset=self._col_names).reset_index(drop=True)

        self._df     = df
        self._Teff   = df['Teff'].values.astype(float)
        self._logg   = df['logg'].values.astype(float)
        self._feh    = df['Z'].values.astype(float)
        self._coeffs = [df[c].values.astype(float) for c in self._col_names]

        if self.verbose:
            mod_lbl = 'ATLAS' if self.mod == 'A' else 'PHOENIX'
            met_lbl = 'LSM'   if self.met == 'L' else 'PCM/CFM'
            xi_lbl  = f'  |  xi={self.xi} km/s' if self.mod == 'A' else ''
            print('=' * 62)
            print(f'CLARET 2017  |  TESS  |  {self.law}')
            print(f'Modelo: {mod_lbl}  |  Método: {met_lbl}{xi_lbl}')
            print(f'Coeficientes: {self._col_names}  |  Pontos: {len(self._Teff)}')
            print(f'Teff : [{self._Teff.min():.0f} – {self._Teff.max():.0f}] K')
            print(f'logg : [{self._logg.min():.1f} – {self._logg.max():.1f}]')
            print(f'[Fe/H]: [{self._feh.min():.2f} – {self._feh.max():.2f}]')
            print('=' * 62)

    # ── Interpolador ──────────────────────────────────────────────────────────
    def _build_interpolator(self):
        # Se Z é constante (ex: PHOENIX Z=0), usa interpolador 2D (Teff, logg)
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

    # ── Interpolação ──────────────────────────────────────────────────────────
    def get_coefficients(self, teff, logg, feh):
        """
        Interpola coeficientes LD.

        Parâmetros
        ----------
        teff : float — Temperatura efetiva (K)
        logg : float — log g superficial
        feh  : float — Metalicidade [Fe/H]

        Retorno
        -------
        list[float]
        """
        errs = []
        if not (self.teff_min <= teff <= self.teff_max):
            errs.append(f"Teff={teff:.0f} fora de [{self.teff_min:.0f}, {self.teff_max:.0f}] K")
        if not (self.logg_min <= logg <= self.logg_max):
            errs.append(f"logg={logg:.2f} fora de [{self.logg_min:.1f}, {self.logg_max:.1f}]")
        if not (self.feh_min <= feh <= self.feh_max):
            errs.append(f"[Fe/H]={feh:.2f} fora de [{self.feh_min:.2f}, {self.feh_max:.2f}]")
        if errs:
            raise ValueError(
                "Parâmetros fora da grade Claret 2017:\n  " + "\n  ".join(errs)
            )

        # Grade PHOENIX: Z fixo → interpolação 2D (Teff, logg)
        if self._feh_fixed is not None:
            result = [float(np.atleast_1d(f(teff, logg))[0]) for f in self._interps]
        else:
            result = [float(np.atleast_1d(f(teff, logg, feh))[0]) for f in self._interps]

        if any(np.isnan(v) for v in result):
            raise ValueError(
                f"Interpolação retornou NaN para Teff={teff}, logg={logg}, [Fe/H]={feh}.\n"
                f"O ponto pode estar em região sem cobertura da grade."
            )
        return result

    # ── Método estático chamado pelo __init__.py ──────────────────────────────
    @staticmethod
    def interpolate(teff, logg, feh, passband='TESS', law='quadratic',
                    xi=2.0, met='L', mod='A', verbose=False, **kwargs):
        """
        Interface estática para chamada pelo meidem.__init__.

        Retorno
        -------
        tuple(list[float], dict) — (coeficientes, metadados)
        """
        grid   = Claret2017Grid(law=law, xi=xi, met=met, mod=mod, verbose=verbose)
        coeffs = grid.get_coefficients(teff, logg, feh)
        meta   = {
            'col_names': grid._col_names,
            'xi'       : xi if mod.upper() == 'A' else None,
            'met'      : met,
            'mod'      : mod,
            'teff_min' : grid.teff_min, 'teff_max': grid.teff_max,
            'logg_min' : grid.logg_min, 'logg_max': grid.logg_max,
            'feh_min'  : grid.feh_min,  'feh_max' : grid.feh_max,
        }
        return coeffs, meta