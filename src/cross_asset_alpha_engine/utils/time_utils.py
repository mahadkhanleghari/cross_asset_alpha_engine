"""Time utilities for Cross-Asset Alpha Engine.

This module provides utilities for handling time zones, trading sessions,
and timestamp alignment for financial data.
"""

import pandas as pd
import pytz
from datetime import datetime, date, time, timedelta
from typing import Optional, Tuple, List, Union

from ..config import TIMEZONE, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE


def normalize_timezone(
    dt: Union[datetime, pd.Timestamp, pd.Series],
    target_tz: str = TIMEZONE
) -> Union[datetime, pd.Timestamp, pd.Series]:
    """Normalize datetime to target timezone.
    
    Args:
        dt: Datetime object, Timestamp, or Series to normalize
        target_tz: Target timezone string
        
    Returns:
        Timezone-normalized datetime object
    """
    target_tz_obj = pytz.timezone(target_tz)
    
    if isinstance(dt, pd.Series):
        # Handle Series of timestamps
        if dt.dt.tz is None:
            # Assume UTC if no timezone
            dt = dt.dt.tz_localize('UTC')
        return dt.dt.tz_convert(target_tz_obj)
    
    elif isinstance(dt, (datetime, pd.Timestamp)):
        # Handle single datetime/timestamp
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(target_tz_obj)
    
    else:
        raise TypeError(f"Unsupported type for timezone normalization: {type(dt)}")


def is_market_hours(
    dt: Union[datetime, pd.Timestamp],
    market_tz: str = TIMEZONE,
    market_open: time = None,
    market_close: time = None
) -> bool:
    """Check if datetime falls within market hours.
    
    Args:
        dt: Datetime to check
        market_tz: Market timezone
        market_open: Market open time (uses config default if None)
        market_close: Market close time (uses config default if None)
        
    Returns:
        True if within market hours, False otherwise
    """
    if market_open is None:
        market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    
    if market_close is None:
        market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    
    # Normalize to market timezone
    dt_normalized = normalize_timezone(dt, market_tz)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if dt_normalized.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if within trading hours
    current_time = dt_normalized.time()
    return market_open <= current_time <= market_close


def get_trading_sessions(
    start_date: date,
    end_date: date,
    market_tz: str = TIMEZONE,
    market_open: time = None,
    market_close: time = None
) -> List[Tuple[datetime, datetime]]:
    """Get list of trading session start/end times.
    
    Args:
        start_date: Start date for sessions
        end_date: End date for sessions
        market_tz: Market timezone
        market_open: Market open time (uses config default if None)
        market_close: Market close time (uses config default if None)
        
    Returns:
        List of (session_start, session_end) tuples
    """
    if market_open is None:
        market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    
    if market_close is None:
        market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    
    tz = pytz.timezone(market_tz)
    sessions = []
    
    current_date = start_date
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            session_start = tz.localize(
                datetime.combine(current_date, market_open)
            )
            session_end = tz.localize(
                datetime.combine(current_date, market_close)
            )
            sessions.append((session_start, session_end))
        
        current_date += timedelta(days=1)
    
    return sessions


def align_timestamps(
    df: pd.DataFrame,
    freq: str = "1min",
    method: str = "ffill",
    market_hours_only: bool = True
) -> pd.DataFrame:
    """Align DataFrame timestamps to regular frequency.
    
    Args:
        df: DataFrame with timestamp index or column
        freq: Target frequency (e.g., '1min', '5min', '1H', '1D')
        method: Fill method for missing values ('ffill', 'bfill', 'interpolate')
        market_hours_only: Whether to include only market hours
        
    Returns:
        DataFrame with aligned timestamps
    """
    # Ensure we have a datetime index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index or 'timestamp' column")
    
    # Create regular frequency index
    start_time = df.index.min()
    end_time = df.index.max()
    
    if market_hours_only and freq.endswith('min'):
        # Create index with only market hours
        regular_index = []
        sessions = get_trading_sessions(
            start_time.date(),
            end_time.date()
        )
        
        for session_start, session_end in sessions:
            session_index = pd.date_range(
                start=session_start,
                end=session_end,
                freq=freq
            )
            regular_index.extend(session_index)
        
        regular_index = pd.DatetimeIndex(regular_index).sort_values()
    else:
        # Create regular index for entire period
        regular_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq=freq
        )
    
    # Reindex DataFrame
    df_aligned = df.reindex(regular_index)
    
    # Fill missing values
    if method == "ffill":
        df_aligned = df_aligned.fillna(method="ffill")
    elif method == "bfill":
        df_aligned = df_aligned.fillna(method="bfill")
    elif method == "interpolate":
        df_aligned = df_aligned.interpolate()
    
    return df_aligned


def resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    price_cols: List[str] = None,
    volume_col: str = "volume"
) -> pd.DataFrame:
    """Resample OHLCV data to different frequency.
    
    Args:
        df: DataFrame with OHLCV data
        freq: Target frequency (e.g., '5min', '1H', '1D')
        price_cols: List of price column names
        volume_col: Volume column name
        
    Returns:
        Resampled DataFrame
    """
    if price_cols is None:
        price_cols = ["open", "high", "low", "close"]
    
    # Ensure we have a datetime index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index or 'timestamp' column")
    
    # Define aggregation rules
    agg_rules = {}
    
    if "open" in price_cols and "open" in df.columns:
        agg_rules["open"] = "first"
    
    if "high" in price_cols and "high" in df.columns:
        agg_rules["high"] = "max"
    
    if "low" in price_cols and "low" in df.columns:
        agg_rules["low"] = "min"
    
    if "close" in price_cols and "close" in df.columns:
        agg_rules["close"] = "last"
    
    if volume_col in df.columns:
        agg_rules[volume_col] = "sum"
    
    if "vwap" in df.columns:
        # VWAP needs special handling - volume-weighted average
        df["vwap_volume"] = df["vwap"] * df[volume_col]
        agg_rules["vwap_volume"] = "sum"
    
    # Resample
    resampled = df.resample(freq).agg(agg_rules)
    
    # Recalculate VWAP if present
    if "vwap" in df.columns:
        resampled["vwap"] = resampled["vwap_volume"] / resampled[volume_col]
        resampled = resampled.drop("vwap_volume", axis=1)
    
    # Drop rows with no data
    resampled = resampled.dropna(how="all")
    
    return resampled


def get_business_days(
    start_date: date,
    end_date: date,
    holidays: Optional[List[date]] = None
) -> List[date]:
    """Get list of business days between start and end dates.
    
    Args:
        start_date: Start date
        end_date: End date
        holidays: List of holiday dates to exclude
        
    Returns:
        List of business day dates
    """
    business_days = pd.bdate_range(start=start_date, end=end_date).date
    
    if holidays:
        business_days = [d for d in business_days if d not in holidays]
    
    return list(business_days)


def calculate_time_to_market_close(
    dt: Union[datetime, pd.Timestamp],
    market_tz: str = TIMEZONE,
    market_close: time = None
) -> timedelta:
    """Calculate time remaining until market close.
    
    Args:
        dt: Current datetime
        market_tz: Market timezone
        market_close: Market close time (uses config default if None)
        
    Returns:
        Time remaining until market close
    """
    if market_close is None:
        market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    
    # Normalize to market timezone
    dt_normalized = normalize_timezone(dt, market_tz)
    
    # Get market close time for the same day
    close_dt = dt_normalized.replace(
        hour=market_close.hour,
        minute=market_close.minute,
        second=0,
        microsecond=0
    )
    
    # If current time is after market close, get next business day's close
    if dt_normalized.time() > market_close or dt_normalized.weekday() >= 5:
        # Find next business day
        next_day = dt_normalized.date() + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        
        close_dt = dt_normalized.replace(
            year=next_day.year,
            month=next_day.month,
            day=next_day.day,
            hour=market_close.hour,
            minute=market_close.minute,
            second=0,
            microsecond=0
        )
    
    return close_dt - dt_normalized


def get_market_calendar(
    start_date: date,
    end_date: date,
    market_tz: str = TIMEZONE
) -> pd.DataFrame:
    """Get market calendar with trading sessions.
    
    Args:
        start_date: Start date
        end_date: End date
        market_tz: Market timezone
        
    Returns:
        DataFrame with market calendar information
    """
    sessions = get_trading_sessions(start_date, end_date, market_tz)
    
    calendar_data = []
    for session_start, session_end in sessions:
        calendar_data.append({
            "date": session_start.date(),
            "session_start": session_start,
            "session_end": session_end,
            "is_trading_day": True
        })
    
    df = pd.DataFrame(calendar_data)
    
    # Add non-trading days
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    trading_dates = set(df["date"])
    
    for dt in all_dates:
        if dt.date() not in trading_dates:
            calendar_data.append({
                "date": dt.date(),
                "session_start": None,
                "session_end": None,
                "is_trading_day": False
            })
    
    df = pd.DataFrame(calendar_data)
    df = df.sort_values("date").reset_index(drop=True)
    
    return df
