"""Uniswap v3 tick and range utilities.

Implements tick arithmetic as:
    p(Delta) = 1.0001^Delta
    tick(p) = log(p) / log(1.0001)

Fee-tier tick spacings:
    10  -> 0.05% fee tier
    60  -> 0.30% fee tier
    200 -> 1.00% fee tier

Custom — Context7 found no library equivalent for Uniswap v3 tick mechanics.
"""
import numpy as np
from typing import Tuple

TICK_BASE = 1.0001
LOG_TICK_BASE = np.log(TICK_BASE)

FEE_TIER_TICK_SPACING = {
    500:   10,   # 0.05%
    3000:  60,   # 0.30%
    10000: 200,  # 1.00%
}


def price_to_tick(price: float) -> float:
    """Convert price to tick (continuous, not rounded).
    
    tick = log(price) / log(1.0001)
    
    Args:
        price: Token X price > 0.
    
    Returns:
        Continuous tick value.
    """
    if price <= 0:
        raise ValueError('price must be positive.')
    return np.log(price) / LOG_TICK_BASE


def tick_to_price(tick: float) -> float:
    """Convert tick to price.
    
    price = 1.0001^tick
    
    Args:
        tick: Tick value (integer or float).
    
    Returns:
        Price corresponding to the tick.
    """
    return TICK_BASE ** tick


def usable_tick_lower(tick: float, spacing: int) -> int:
    """Floor tick to nearest usable lower tick (multiple of spacing).
    
    Args:
        tick: Raw tick value.
        spacing: Tick spacing for the fee tier.
    
    Returns:
        Largest usable tick <= tick.
    """
    return int(np.floor(tick / spacing)) * spacing


def usable_tick_upper(tick: float, spacing: int) -> int:
    """Ceiling tick to nearest usable upper tick (multiple of spacing).
    
    Args:
        tick: Raw tick value.
        spacing: Tick spacing for the fee tier.
    
    Returns:
        Smallest usable tick >= tick.
    """
    return int(np.ceil(tick / spacing)) * spacing


def build_unitary_range(p_b_target: float, spot: float, spacing: int) -> Tuple[float, float]:
    """Build a unitary range [p_a, p_b) spanning one tick interval.
    
    Maps p_b_target to the nearest usable upper tick compatible with
    the fee-tier spacing, then sets p_a to the adjacent lower usable
    tick, yielding exactly one tick interval.
    
    Args:
        p_b_target: Target upper price (e.g., M * S_0 where M is moneyness).
        spot:       Current spot price S_0 (unused directly but validates M).
        spacing:    Tick spacing for the chosen fee tier.
    
    Returns:
        Tuple (p_a, p_b) where p_a and p_b are tick-aligned prices.
    """
    tick_b_raw = price_to_tick(p_b_target)
    tick_b = usable_tick_upper(tick_b_raw, spacing)
    tick_a = tick_b - spacing  # adjacent lower usable tick
    p_b = tick_to_price(tick_b)
    p_a = tick_to_price(tick_a)
    return p_a, p_b


def moneyness_grid(spot: float, m_min: float, m_max: float, n_points: int) -> np.ndarray:
    """Generate a moneyness grid M = p_b / S_0.
    
    Args:
        spot:     Current spot price S_0.
        m_min:    Minimum moneyness (e.g., 0.8).
        m_max:    Maximum moneyness (e.g., 1.5).
        n_points: Number of moneyness values.
    
    Returns:
        Array of p_b_target values (= M * spot).
    """
    M = np.linspace(m_min, m_max, n_points)
    return M * spot
