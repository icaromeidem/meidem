"""
meidem/grids/kostogryz2022.py
==============================
Interpolador de coeficientes de limb darkening
Grade: Kostogryz et al. (2022), MPS-ATLAS

Tabelas (parquet):
  table5.parquet — lei nonlinear (4 coeficientes): a1, a2, a3, a4
  table6.parquet — lei power-2: c, alpha

Passbands disponíveis: TESS, Kepler, CHEOPS, PLATO

Referência:
  Kostogryz, N. V. et al. (2022)
  https://doi.org/10.1051/0004-6361/202140376
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# Caminho para as tabelas dentro do pacote
_TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tables', 'kostogryz2022')

# Mapa lei → arquivo e colunas de coeficientes
_LAW_MAP = {
    'nonlinear': ('table5.parquet', ['a1', 'a2', 'a3', 'a4']),
    'power2'   : ('table6.parquet', ['c', 'alpha']),
}

_VALID_PASSBANDS = ['TESS', 'Kepler', 'CHEOPS', 'PLATO']
_VALID_LAWS      = list(_LAW_MAP.keys())


class KostogryzGrid:
    """
    Interpolador tri-linear (Teff, logg, [Fe/H]) para coeficientes de LD
    da grade MPS-ATLAS de Kostogryz et al. (2022).

    Parâmetros
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
                f"passband='{self.passband}' não suportado para Kostogryz+2022.\n"
                f"Opções: {_VALID_PASSBANDS}"
            )
        if self.law not in _LAW_MAP:
            raise ValueError(
                f"law='{self.law}' não suportada para Kostogryz+2022.\n"
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

        # A tabela Kostogryz tem coluna 'passband' — filtra
        if 'passband' in df.columns:
            df = df[df['passband'] == self.passband].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(
                    f"Passband '{self.passband}' não encontrada na tabela.\n"
                    f"Disponíveis: {_VALID_PASSBANDS}"
                )

        # Verifica colunas necessárias
        required = ['Teff', 'logg', 'M/H'] + self._col_names
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Colunas ausentes na tabela Kostogryz+2022: {missing}\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

        # Remove NaN nos coeficientes
        df = df.dropna(subset=self._col_names).reset_index(drop=True)

        self._df   = df
        self._Teff = df['Teff'].values.astype(float)
        self._logg = df['logg'].values.astype(float)
        self._feh  = df['M/H'].values.astype(float)
        self._coeffs = [df[c].values.astype(float) for c in self._col_names]

        if self.verbose:
            print('=' * 62)
            print(f'KOSTOGRYZ+2022  |  MPS-ATLAS  |  {self.passband}  |  {self.law}')
            print(f'Coeficientes: {self._col_names}  |  Pontos: {len(self._Teff)}')
            print(f'Teff : [{self._Teff.min():.0f} – {self._Teff.max():.0f}] K')
            print(f'logg : [{self._logg.min():.1f} – {self._logg.max():.1f}]')
            print(f'[Fe/H]: [{self._feh.min():.2f} – {self._feh.max():.2f}]')
            print('=' * 62)

    # ── Interpolador ──────────────────────────────────────────────────────────
    def _build_interpolator(self):
        pts = np.column_stack((self._Teff, self._logg, self._feh))
        self._interps = [LinearNDInterpolator(pts, c) for c in self._coeffs]
        self.teff_min, self.teff_max = self._Teff.min(), self._Teff.max()
        self.logg_min, self.logg_max = self._logg.min(), self._logg.max()
        self.feh_min,  self.feh_max  = self._feh.min(),  self._feh.max()

    # ── Interpolação ──────────────────────────────────────────────────────────
    def get_coefficients(self, teff, logg, feh):
        """
        Interpola coeficientes LD para os parâmetros estelares fornecidos.

        Parâmetros
        ----------
        teff : float — Temperatura efetiva (K)
        logg : float — log g superficial
        feh  : float — Metalicidade [Fe/H]

        Retorno
        -------
        list[float] — coeficientes interpolados
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
                "Parâmetros fora da grade Kostogryz+2022:\n  " + "\n  ".join(errs)
            )

        result = [float(np.atleast_1d(f(teff, logg, feh))[0]) for f in self._interps]

        if any(np.isnan(v) for v in result):
            raise ValueError(
                f"Interpolação retornou NaN para Teff={teff}, logg={logg}, [Fe/H]={feh}.\n"
                f"O ponto pode estar em região sem cobertura da grade."
            )
        return result

    # ── Método estático chamado pelo __init__.py ──────────────────────────────
    @staticmethod
    def interpolate(teff, logg, feh, passband='TESS', law='power2', verbose=False, **kwargs):
        """
        Interface estática para chamada pelo meidem.__init__.

        Retorno
        -------
        tuple(list[float], dict) — (coeficientes, metadados)
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
