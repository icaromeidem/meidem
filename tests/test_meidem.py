"""
tests/test_meidem.py
====================
Smoke tests for MEIDEM — verifies that all grids return physically
reasonable LD coefficients for the Sun (IAU 2015 parameters).

These are not precision tests — they check that:
  1. The function runs without errors
  2. The correct number of coefficients is returned
  3. Coefficients are finite and within physically plausible bounds
  4. The return dict has all expected keys
  5. Grid-specific metadata is correctly populated

Run with:
    pytest tests/test_meidem.py -v
"""

import warnings
import math
import pytest
import meidem

# ── IAU 2015 nominal solar parameters ─────────────────────────────────────────
SUN = dict(teff=5772, logg=4.438, feh=0.0)

# Cool star for PHOENIX tests
COOL_STAR = dict(teff=3800, logg=4.8, feh=0.0)

# ── Expected return dict keys ─────────────────────────────────────────────────
EXPECTED_KEYS = {
    'coefficients', 'n_coeffs', 'law', 'passband', 'grid',
    'reference', 'doi', 'teff_input', 'logg_input', 'feh_input',
    'xi', 'met', 'mod',
}


def _check_result(result, n_coeffs_expected, grid, passband, law, star=None):
    """Shared assertions for all grid tests."""
    if star is None:
        star = SUN

    # All keys present
    assert EXPECTED_KEYS == set(result.keys()), \
        f"Missing or extra keys: {set(result.keys()) ^ EXPECTED_KEYS}"

    # Correct number of coefficients
    assert result['n_coeffs'] == n_coeffs_expected
    assert len(result['coefficients']) == n_coeffs_expected

    # All coefficients are finite real numbers
    for i, c in enumerate(result['coefficients']):
        assert math.isfinite(c), f"Coefficient [{i}] is not finite: {c}"

    # Coefficients are within physically plausible bounds (-2, 2)
    for i, c in enumerate(result['coefficients']):
        assert -2.0 < c < 2.0, f"Coefficient [{i}]={c:.4f} outside plausible range (-2, 2)"

    # Metadata
    assert result['grid']     == grid
    assert result['passband'] == passband
    assert result['law']      == law
    assert result['teff_input'] == star['teff']
    assert result['logg_input'] == star['logg']
    assert result['feh_input']  == star['feh']
    assert isinstance(result['reference'], str) and len(result['reference']) > 0
    assert isinstance(result['doi'], str) and result['doi'].startswith('10.')


# ── kostogryz2022 ─────────────────────────────────────────────────────────────

class TestKostogryz2022:

    def test_power2_tess(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='kostogryz2022', law='power2')
        _check_result(r, n_coeffs_expected=2, grid='kostogryz2022',
                      passband='TESS', law='power2')
        assert r['xi']  is None
        assert r['met'] is None
        assert r['mod'] is None

    def test_nonlinear_tess(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='kostogryz2022', law='nonlinear')
        _check_result(r, n_coeffs_expected=4, grid='kostogryz2022',
                      passband='TESS', law='nonlinear')

    def test_power2_kepler(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='Kepler', grid='kostogryz2022', law='power2')
        _check_result(r, n_coeffs_expected=2, grid='kostogryz2022',
                      passband='Kepler', law='power2')

    def test_power2_cheops(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='CHEOPS', grid='kostogryz2022', law='power2')
        _check_result(r, n_coeffs_expected=2, grid='kostogryz2022',
                      passband='CHEOPS', law='power2')

    def test_default_law_is_power2(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='kostogryz2022')
        assert r['law'] == 'power2'

    def test_out_of_grid_raises(self):
        with pytest.raises(ValueError, match="outside"):
            meidem.get_ld_coefficients(
                teff=99999, logg=4.44, feh=0.0,
                passband='TESS', grid='kostogryz2022')

    def test_invalid_passband_raises(self):
        with pytest.raises(ValueError):
            meidem.get_ld_coefficients(
                **SUN, passband='INVALID', grid='kostogryz2022')

    def test_invalid_law_raises(self):
        with pytest.raises(ValueError):
            meidem.get_ld_coefficients(
                **SUN, passband='TESS', grid='kostogryz2022', law='invalid')


# ── claret2022 ────────────────────────────────────────────────────────────────

class TestClaret2022:

    def test_power2_tess(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='claret2022', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2022',
                      passband='TESS', law='power2')
        assert r['xi'] == 2.0

    def test_power2_kepler(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='Kepler', grid='claret2022', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2022',
                      passband='Kepler', law='power2')

    def test_power2_gaia_g(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='Gaia_G', grid='claret2022', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2022',
                      passband='Gaia_G', law='power2')

    def test_power2_sdss_r(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='SDSS_r', grid='claret2022', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2022',
                      passband='SDSS_r', law='power2')

    def test_power2_johnson_v(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='Johnson_V', grid='claret2022', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2022',
                      passband='Johnson_V', law='power2')

    def test_xi_warning_when_default(self):
        with pytest.warns(UserWarning, match="xi=2.0"):
            meidem.get_ld_coefficients(
                **SUN, passband='TESS', grid='claret2022')

    def test_no_warning_when_xi_explicit(self):
        # passing xi=2.0 explicitly must NOT trigger a UserWarning
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            meidem.get_ld_coefficients(
                **SUN, passband='TESS', grid='claret2022', xi=2.0)
        user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0, \
            f"Unexpected UserWarning when xi is set explicitly: {[str(w.message) for w in user_warnings]}"

    def test_xi_in_return_dict(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='claret2022', xi=2.0)
        assert r['xi'] == 2.0


# ── claret2017 ────────────────────────────────────────────────────────────────

class TestClaret2017:

    def test_quadratic_atlas_lsm(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='claret2017',
            law='quadratic', mod='A', met='L', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2017',
                      passband='TESS', law='quadratic')
        assert r['xi']  == 2.0
        assert r['met'] == 'L'
        assert r['mod'] == 'A'

    def test_quadratic_atlas_pcm(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='claret2017',
            law='quadratic', mod='A', met='F', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2017',
                      passband='TESS', law='quadratic')
        assert r['met'] == 'F'

    def test_4coeff_atlas(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='claret2017',
            law='4coeff', mod='A', met='L', xi=2.0)
        _check_result(r, n_coeffs_expected=4, grid='claret2017',
                      passband='TESS', law='4coeff')

    def test_quadratic_phoenix_cool_star(self):
        r = meidem.get_ld_coefficients(
            **COOL_STAR, passband='TESS', grid='claret2017',
            law='quadratic', mod='P')
        _check_result(r, n_coeffs_expected=2, grid='claret2017',
                      passband='TESS', law='quadratic', star=COOL_STAR)
        assert r['xi']  is None   # PHOENIX has no xi
        assert r['mod'] == 'P'

    def test_xi_none_for_phoenix(self):
        r = meidem.get_ld_coefficients(
            **COOL_STAR, passband='TESS', grid='claret2017',
            law='quadratic', mod='P')
        assert r['xi'] is None

    def test_default_law_is_quadratic(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='claret2017', xi=2.0)
        assert r['law'] == 'quadratic'

    def test_y_law_atlas_only(self):
        with pytest.raises(ValueError):
            meidem.get_ld_coefficients(
                **SUN, passband='TESS', grid='claret2017',
                law='y', mod='P')


# ── claret2011 ────────────────────────────────────────────────────────────────

class TestClaret2011:

    def test_quadratic_kepler_atlas(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='Kp', grid='claret2011',
            law='quadratic', mod='A', met='L', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2011',
                      passband='Kp', law='quadratic')
        assert r['xi']  == 2.0
        assert r['met'] == 'L'
        assert r['mod'] == 'A'

    def test_quadratic_corot(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='C', grid='claret2011',
            law='quadratic', mod='A', met='L', xi=2.0)
        _check_result(r, n_coeffs_expected=2, grid='claret2011',
                      passband='C', law='quadratic')

    def test_4coeff_kepler(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='Kp', grid='claret2011',
            law='4coeff', mod='A', met='L', xi=2.0)
        _check_result(r, n_coeffs_expected=4, grid='claret2011',
                      passband='Kp', law='4coeff')

    def test_default_law_is_quadratic(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='Kp', grid='claret2011', xi=2.0)
        assert r['law'] == 'quadratic'


# ── API general ───────────────────────────────────────────────────────────────

class TestAPI:

    def test_invalid_grid_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            meidem.get_ld_coefficients(
                **SUN, passband='TESS', grid='invalid_grid')

    def test_grid_name_case_insensitive(self):
        r = meidem.get_ld_coefficients(
            **SUN, passband='TESS', grid='Kostogryz2022', law='power2')
        assert r['grid'] == 'kostogryz2022'

    def test_available_grids_returns_list(self):
        grids = meidem.available_grids(verbose=False)
        assert isinstance(grids, list)
        assert 'kostogryz2022' in grids
        assert 'claret2022'    in grids
        assert 'claret2017'    in grids
        assert 'claret2011'    in grids

    def test_available_laws_returns_list(self):
        laws = meidem.available_laws('claret2017', verbose=False)
        assert isinstance(laws, list)
        assert 'quadratic' in laws
        assert '4coeff'    in laws

    def test_available_passbands_returns_list(self):
        pbs = meidem.available_passbands('claret2022', verbose=False)
        assert isinstance(pbs, list)
        assert 'TESS'   in pbs
        assert 'Gaia_G' in pbs

    def test_available_laws_invalid_grid_raises(self):
        with pytest.raises(ValueError):
            meidem.available_laws('invalid_grid')

    def test_available_passbands_invalid_grid_raises(self):
        with pytest.raises(ValueError):
            meidem.available_passbands('invalid_grid')