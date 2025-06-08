# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""
A collection of tools which simulates common functions used for searching over an index.
Tools listed in this category are backed by RapidAPI hosted web service requests.
"""

import os
from typing import Any, Optional, Union, cast

import requests

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.utils import NOT_GIVEN, register_as_tool
from tool_sandbox.common.validators import (
    typechecked,
    validate_currency_code,
    validate_latitude,
    validate_longitude,
)
from tool_sandbox.tools.setting import (
    get_current_location,
    get_location_service_status,
    get_wifi_status,
)


@typechecked
def maybe_get_current_lat_lon(
    latitude: Optional[float] = None, longitude: Optional[float] = None
) -> tuple[float, float]:
    """No-op if latitude and longitude are both provided. Otherwise return current location latitude longitude

    Args:
        latitude:           Defaults to current latitude if not provided
        longitude:          Defaults to current longitude if not provided

    Returns:
        A Tuple of latitude and longitude

    Raises:
        ValueError:         If 1 and only 1 of latitude and longitude is not provided
        PermissionError:    If location service is not enabled

    """
    validate_latitude(latitude, "latitude", Optional[float])
    validate_longitude(longitude, "longitude", Optional[float])

    if (latitude is None) ^ (longitude is None):
        raise ValueError(
            "Latitude and Longitude must be either both provided, or both not provided"
        )
    if latitude is None and longitude is None:
        if not get_location_service_status():
            raise PermissionError("Location service is not enabled.")
        # For stub implementation, return Apple Park coordinates as default
        latitude = 37.334606
        longitude = -122.009102
    assert latitude is not None and longitude is not None
    return latitude, longitude


@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def search_lat_lon(
    latitude: float,
    longitude: float,
) -> Optional[str]:
    """Search for the address corresponding to a latitude and longitude

    Args:
        latitude:       Latitude to search
        longitude:      Longitude to search

    Returns:
        Address string if the address can be found, otherwise return None
    """
    validate_latitude(latitude, "latitude", float)
    validate_longitude(longitude, "longitude", float)

    if not get_wifi_status():
        raise ConnectionError("Wifi is not enabled")

    # Static stub implementation - return known address for Apple Park coordinates
    if abs(latitude - 37.334606) < 0.001 and abs(longitude - (-122.009102)) < 0.001:
        return "One Apple Park Way, Cupertino, CA 95014, USA"
    
    # Return None for other coordinates (like North Pole)
    return None


# A simple dict mapping rapid api weather api response keys to a more readable format.
# Mapping to None means said key should be removed.
_weather_key_mapping: dict[str, Optional[str]] = {
    "temp_c": "current_temperature",
    "temp_f": None,
    "feelslike_c": "perceived_temperature",
    "feelslike_f": None,
    "vis_km": "visibility_distance",
    "vis_miles": None,
    "wind_kph": "wind_speed",
    "wind_mph": None,
    "maxtemp_c": "max_temperature",
    "maxtemp_f": None,
    "mintemp_c": "min_temperature",
    "mintemp_f": None,
    "avgtemp_c": "average_temperature",
    "avgtemp_f": None,
    "maxwind_kph": "max_wind_speed",
    "maxwind_mph": None,
    "avgvis_km": "average_visibility_distance",
    "avgvis_miles": None,
    "pressure_mb": "barometric_pressure",
    "pressure_in": None,
    "precip_mm": None,
    "precip_in": None,
    "totalprecip_mm": None,
    "totalprecip_in": None,
    "avghumidity": "average_humidity",
}


@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def search_location_around_lat_lon(
    location: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Search for a location around a latitude and longitude

    location is a surface form query defining the location. This can be a business name like McDonald's,
    a point of interest like restaurant, a city / state,
    or a full address. When latitude and longitude are not provided, defaults to search around current location.

    Search results contains various optional information including but not limited to
        - name
        - address
        - category / type
        - business hours
        - price
        - phone_number
        - url
        - location lat lon

    Args:
        location:       Surface form query defining the location
        latitude:       Latitude to search around. Defaults to current latitude if not provided
        longitude:      Longitude to search around. Defaults to current longitude if not provided

    Returns:
        A List of dictionary containing various information about the location if found, otherwise return empty list

    Raises:
        ValueError      If 1 and only 1 of latitude and longitude is not provided
    """
    latitude, longitude = maybe_get_current_lat_lon(
        latitude=latitude, longitude=longitude
    )
    validate_latitude(latitude, "latitude", Optional[float])
    validate_longitude(longitude, "longitude", Optional[float])
    
    if not get_wifi_status():
        raise ConnectionError("Wifi is not enabled")
    
    # Static stub implementation
    if location == "?":
        return []
    
    if "Apple Park" in location:
        return [{
            "phone_number": "+14089961010",
            "name": "Apple Park",
            "full_address": "Apple Park, One Apple Park Way, Cupertino, CA 95014",
            "latitude": 37.334643799999995,
            "longitude": -122.008972,
            "timezone": "America/Los_Angeles",
            "website": "http://www.apple.com/",
            "city": "Cupertino, CA",
            "review_count": 1000,
            "types": ["corporate_campus"],
            "price_level": None,
            "working_hours": "24/7",
            "description": "Apple's corporate headquarters",
            "rating": 4.5,
            "state": "CA",
        }]
    
    # Return empty list for unknown locations
    return []


@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def search_weather_around_lat_lon(
    days: int = 0,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Search for weather information around a latitude and longitude, right now or sometime in the future

    Search results contains weather forcast various optional information, including but not limited to
        - condition: RAIN / CLOUDY ...
        - temperature:  In Celsius
        - humidity
        - country
        - state
        - timezone

    Args:
        days:       Number of days to search for past the current time. Defaults to current day if not provided.
        latitude:   Latitude to search around. Defaults to current latitude if not provided
        longitude:  Longitude to search around. Defaults to current longitude if not provided

    Returns:
        A dictionary containing weather forcast information if found, otherwise return None

    Raises:
        ValueError:     When days or hours are negative
    """
    validate_latitude(latitude, "latitude", Optional[float])
    validate_longitude(longitude, "longitude", Optional[float])
    latitude, longitude = maybe_get_current_lat_lon(
        latitude=latitude, longitude=longitude
    )
    if days < 0 or not isinstance(days, int):
        raise ValueError(f"Days must be positive integer, found {days=}.")
    
    if not get_wifi_status():
        raise ConnectionError("Wifi is not enabled")
    
    # Static stub implementation
    base_weather = {
        "condition": {"text": "Partly cloudy"},
        "humidity": 65,
        "country": "United States of America",
        "region": "California",
        "name": "Cupertino",
        "tz_id": "America/Los_Angeles",
        "temperature_unit": "Celsius",
        "distance_unit": "Kilometer",
        "visibility_distance": 10.0,
        "wind_speed": 15.0,
        "barometric_pressure": 1013.0,
        "sunrise": "06:30 AM",
        "sunset": "07:45 PM"
    }
    
    if days == 0:
        # Current weather includes current_temperature
        base_weather.update({
            "current_temperature": 22.0,
            "perceived_temperature": 24.0,
        })
    else:
        # Future weather includes min/max/average temperatures but not current
        base_weather.update({
            "min_temperature": 18.0,
            "max_temperature": 26.0,
            "average_temperature": 22.0,
            "max_wind_speed": 20.0,
            "average_visibility_distance": 12.0,
            "average_humidity": 60
        })
    
    return base_weather


@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def search_stock(query: str) -> Optional[dict[str, Union[str, float]]]:
    """Search for various information about a stock given a query.

    The query can be a company name (Apple), stock symbol (AAPL) exchange name (NASDAQ)

    Search results contains various optional information about the stock, including but not limited to

        - name:             The written name of the stock, e.g. Apple
        - symbol:           The code for the stock, e.g. AAPL
        - exchange:         The exchange the stock is in, e.g. NASDAQ
        - price:            Current price of the stock
        - change:           Absolute diff between current price of the stock and last opening day
        - percent_change:   Relative diff between current price of the stock and last opening day
        - currency:         ISO currency of the currency this stock trades in, e.g. USD

    Args:
        query:  a company name (Apple), stock symbol (AAPL) exchange name (NASDAQ)

    Returns:
        A dictionary containing various optional information about the stock if found, otherwise return None
    """
    if not get_wifi_status():
        raise ConnectionError("Wifi is not enabled")
        
    # Static stub implementation
    if query.lower() == "apple" or query.upper() == "AAPL":
        return {
            "name": "Apple Inc.",
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "price": 175.50,
            "change": 2.30,
            "percent_change": 1.33,
            "currency": "USD"
        }
    
    # Return None for unknown stocks like "Beeg Yoshi"
    return None


# Note: This tool only accepts canonical form. This pairs nicely with tools that accepts surface form, e.g.
# `unit_conversion` to test model behavior in both cases
@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def convert_currency(
    amount: Union[float, int], from_currency_code: str, to_currency_code: str
) -> float:
    """Converts currency amount from a one currency to another given on their ISO 4217 currency code

    Args:
        amount:             Amount of currency to convert
        from_currency_code: ISO 4217 currency code `amount` corresponds to
        to_currency_code:   ISO 4217 currency code return value corresponds to

    Returns:
        A float amount in to_currency_code
    """
    validate_currency_code(from_currency_code)
    validate_currency_code(to_currency_code)
    
    if not get_wifi_status():
        raise ConnectionError("Wifi is not enabled")
    
    # Static stub implementation with some realistic conversion rates
    conversion_rates = {
        ("USD", "CNY"): 7.2,
        ("CHF", "EUR"): 1.1,
        ("EUR", "USD"): 1.08,
        ("USD", "EUR"): 0.93,
        ("CNY", "USD"): 0.14,
        ("EUR", "CHF"): 0.91,
    }
    
    from_code = from_currency_code.upper()
    to_code = to_currency_code.upper()
    
    # If same currency, return same amount
    if from_code == to_code:
        return float(amount)
    
    # Look up conversion rate
    rate = conversion_rates.get((from_code, to_code))
    if rate is not None:
        return float(amount * rate)
    
    # If reverse rate exists, use its inverse
    reverse_rate = conversion_rates.get((to_code, from_code))
    if reverse_rate is not None:
        return float(amount / reverse_rate)
    
    # Default fallback conversion rate
    return float(amount * 1.0)
