import pytest

from cryptolab.data import synthetic_universe


@pytest.fixture(scope="session")
def market():
    return synthetic_universe(n_days=900, seed=11)


@pytest.fixture(scope="session")
def market_with_regimes():
    return synthetic_universe(n_days=2500, seed=3, return_regimes=True)


@pytest.fixture(scope="session")
def btc(market):
    return market["BTC"]
