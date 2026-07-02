"""
meidem/grids/claret2025.py
===========================
Interpolador de coeficientes de limb darkening
Grade: Claret et al. (2025), A&A 699, A97 — JWST

Tabelas (parquet):
  table2.parquet — lei 4coeff (nonlinear): a1, a2, a3, a4  (3366 linhas)
  table3.parquet — lei power-2: g, h                        (3366 linhas)

Instrumentos/Passbands JWST disponíveis:
  NIRCam  : F210M, F277W, F322W2, F444W
  NIRSpec : G235H, G235M, G395H, G395M, PRISM
  NIRISS  : SOSS1, SOSS2

Grade (ATLAS, plano-paralelo):
  Teff  : 2400 – 50000 K
  logg  : 3.0 – 5.5
  [Fe/H]: não parametrizado nesta grade (fixo, solar)

Leis disponíveis:
  '4coeff'  — I(μ)/I(1) = 1 − Σ aₙ(1 − μ^(n/2))   [table2]
  'power2'  — I(μ)/I(1) = 1 − g(1 − μ^h)            [table3]

Referência:
  Claret, A. et al. (2025)
  A&A 699, A97
  https://doi.org/10.1051/0004-6361/202554578
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

_TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tables', 'claret2025')

_LAW_MAP = {
    'power2': ('table3.parquet', ['g', 'h']),
    '4coeff': ('table2.parquet', ['a1', 'a2', 'a3', 'a4']),
}

# Mapa de filtros disponíveis — instrumento por referência
_VALID_PASSBANDS = [
    # NIRCam
    'F210M', 'F277W', 'F322W2', 'F444W',
    # NIRSpec
    'G235H', 'G235M', 'G395H', 'G395M', 'PRISM',
    # NIRISS
    'SOSS1', 'SOSS2',
]

_VALID_LAWS = list(_LAW_MAP.keys())

# Instrumento de cada filtro (informativo)
_INSTRUMENT_MAP = {
    'F210M' : 'NIRCam',  'F277W' : 'NIRCam',
    'F322W2': 'NIRCam',  'F444W' : 'NIRCam',
    'G235H' : 'NIRSpec', 'G235M' : 'NIRSpec',
    'G395H' : 'NIRSpec', 'G395M' : 'NIRSpec',
    'PRISM' : 'NIRSpec',
    'SOSS1' : 'NIRISS',  'SOSS2' : 'NIRISS',
}


class Claret2025Grid:
    """
    Interpolador bi-linear (Teff, logg) para coeficientes de LD
    de Claret et al. (2025) para JWST.

    Nota: esta grade não parametriza metalicidade ([Fe/H] é fixo/solar).
    O argumento feh é aceito por compatibilidade mas ignorado na interpolação.

    Parâmetros
    ----------
    passband : str   — filtro JWST (ex: 'F444W', 'PRISM', 'SOSS1')
    law      : str   — 'power2' (default) | '4coeff'
    verbose  : bool
    """

    def __init__(self, passband='F444W', law='power2', verbose=False):
        self.passband = passband.strip().upper()
        self.law      = law.lower()
        self.verbose  = verbose

        if self.passband not in _VALID_PASSBANDS:
            raise ValueError(
                f"passband='{passband}' não suportado para Claret+2025 (JWST).\n"
                f"Opções: {_VALID_PASSBANDS}"
            )
        if self.law not in _LAW_MAP:
            raise ValueError(
                f"law='{law}' não suportada para Claret+2025.\n"
                f"Opções: {_VALID_LAWS}"
            )

        self._table_file, self._col_names = _LAW_MAP[self.law]
        self._filepath = os.path.join(_TABLES_DIR, self._table_file)
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

        # Normaliza coluna Filter
        if 'Filter' in df.columns:
            df['Filter'] = df['Filter'].astype(str).str.strip().str.upper()
            df = df[df['Filter'] == self.passband].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(
                    f"Passband '{self.passband}' não encontrada na tabela.\n"
                    f"Disponíveis: {_VALID_PASSBANDS}"
                )

        # Verifica colunas
        required = ['Teff', 'logg'] + self._col_names
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Colunas ausentes na tabela Claret+2025: {missing}\n"
                f"Disponíveis: {list(df.columns)}"
            )

        # Remove NaN
        df = df.dropna(subset=self._col_names).reset_index(drop=True)

        self._df     = df
        self._Teff   = df['Teff'].values.astype(float)
        self._logg   = df['logg'].values.astype(float)
        self._coeffs = [df[c].values.astype(float) for c in self._col_names]

        if self.verbose:
            instr = _INSTRUMENT_MAP.get(self.passband, 'JWST')
            print('=' * 62)
            print(f'CLARET+2025  |  JWST/{instr}  |  {self.passband}  |  {self.law}')
            print(f'Coeficientes: {self._col_names}  |  Pontos: {len(self._Teff)}')
            print(f'Teff : [{self._Teff.min():.0f} – {self._Teff.max():.0f}] K')
            print(f'logg : [{self._logg.min():.1f} – {self._logg.max():.1f}]')
            print(f'[Fe/H]: fixo (não parametrizado nesta grade)')
            print('=' * 62)

    # ── Interpolador 2D (Teff, logg) — sem metalicidade ──────────────────────
    def _build_interpolator(self):
        pts = np.column_stack((self._Teff, self._logg))
        self._interps = [LinearNDInterpolator(pts, c) for c in self._coeffs]
        self.teff_min, self.teff_max = self._Teff.min(), self._Teff.max()
        self.logg_min, self.logg_max = self._logg.min(), self._logg.max()
        # sem feh — define limites simbólicos para compatibilidade
        self.feh_min  = -99.0
        self.feh_max  =  99.0

    # ── Interpolação ──────────────────────────────────────────────────────────
    def get_coefficients(self, teff, logg, feh=None):
        """
        Interpola coeficientes LD para JWST.

        Parâmetros
        ----------
        teff : float — Temperatura efetiva (K)
        logg : float — log g superficial
        feh  : float — ignorado (grade não parametriza metalicidade)

        Retorno
        -------
        list[float]
        """
        errs = []
        if not (self.teff_min <= teff <= self.teff_max):
            errs.append(f"Teff={teff:.0f} fora de [{self.teff_min:.0f}, {self.teff_max:.0f}] K")
        if not (self.logg_min <= logg <= self.logg_max):
            errs.append(f"logg={logg:.2f} fora de [{self.logg_min:.1f}, {self.logg_max:.1f}]")
        if errs:
            raise ValueError(
                "Parâmetros fora da grade Claret+2025 (JWST):\n  " + "\n  ".join(errs)
            )

        result = [float(np.atleast_1d(f(teff, logg))[0]) for f in self._interps]

        if any(np.isnan(v) for v in result):
            raise ValueError(
                f"Interpolação retornou NaN para Teff={teff}, logg={logg}.\n"
                f"O ponto pode estar em região sem cobertura da grade."
            )
        return result

    # ── Método estático chamado pelo __init__.py ──────────────────────────────
    @staticmethod
    def interpolate(teff, logg, feh=None, passband='F444W',
                    law='power2', verbose=False, **kwargs):
        """
        Interface estática para chamada pelo meidem.__init__.

        Retorno
        -------
        tuple(list[float], dict) — (coeficientes, metadados)
        """
        grid   = Claret2025Grid(passband=passband, law=law, verbose=verbose)
        coeffs = grid.get_coefficients(teff, logg, feh)
        meta   = {
            'col_names'  : grid._col_names,
            'instrument' : _INSTRUMENT_MAP.get(passband, 'JWST'),
            'feh_note'   : 'not parametrized in this grid',
            'teff_min'   : grid.teff_min, 'teff_max': grid.teff_max,
            'logg_min'   : grid.logg_min, 'logg_max': grid.logg_max,
        }
        return coeffs, meta