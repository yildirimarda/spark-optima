# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Opt-in live pricing clients with file caching and graceful fallback.

This module provides the optional "live" pricing layer on top of the static
baseline rates and curated regional multipliers in
:mod:`spark_optima.platforms.pricing`. It is **disabled by default** and only
activates when the ``SPARK_OPTIMA_LIVE_PRICING`` environment variable is
truthy (``1``/``true``/``yes``/``on``, case-insensitive).

Supported platforms:

| Platform        | Live source                                  | Rate            |
|-----------------|----------------------------------------------|-----------------|
| ``azure_synapse`` | Azure Retail Prices API (public, no auth)  | vCore-hour      |
| ``aws_emr``     | AWS Pricing API (boto3, credentials needed)  | EC2 on-demand/h |
| ``aws_glue``    | AWS Pricing API (boto3, credentials needed)  | DPU-hour        |

``gcp_dataproc`` and ``databricks`` deliberately stay on static pricing:

- GCP's Cloud Billing Catalog API requires an API key, so it cannot be
  queried anonymously (deferred to the v1.5+ backlog; see PLAN.md).
- Databricks DBU rates are proprietary list prices with no public pricing
  API (and they vary by workspace tier and contract), so there is nothing
  to query.

Resolved rates are cached in a JSON file (default
``~/.spark_optima/pricing_cache.json``, override with
``SPARK_OPTIMA_PRICING_CACHE``) for 24 hours (override with
``SPARK_OPTIMA_PRICING_TTL_HOURS``). Failed lookups are cached as negative
entries for one hour so a dead network does not add latency to every cost
estimate.

The only public entry point used by the platform adapters is
:func:`get_live_hourly_rate`, which returns ``None`` on **any** failure
(disabled flag, unsupported platform, timeout, parse error, missing boto3,
missing credentials, cache corruption) — it never raises. Adapters fall back
to the static baseline rate x regional multiplier whenever it returns
``None``.

Example:
    >>> import os
    >>> os.environ["SPARK_OPTIMA_LIVE_PRICING"] = "1"
    >>> from spark_optima.platforms.live_pricing import get_live_hourly_rate
    >>> get_live_hourly_rate("azure_synapse", region="westeurope")  # doctest: +SKIP
    0.163

"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is normally available
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Environment variables controlling the live pricing layer
LIVE_PRICING_ENV_VAR = "SPARK_OPTIMA_LIVE_PRICING"
CACHE_PATH_ENV_VAR = "SPARK_OPTIMA_PRICING_CACHE"
CACHE_TTL_ENV_VAR = "SPARK_OPTIMA_PRICING_TTL_HOURS"

# Defaults
DEFAULT_CACHE_PATH = Path.home() / ".spark_optima" / "pricing_cache.json"
DEFAULT_TTL_HOURS = 24.0
# Failed lookups are remembered for a short window so an unreachable pricing
# API does not add a network timeout to every single cost estimate.
NEGATIVE_TTL_HOURS = 1.0
# Live pricing must never noticeably slow down cost estimation
REQUEST_TIMEOUT_SECONDS = 5.0

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

# Platforms with a working live pricing source. gcp_dataproc and databricks
# are intentionally absent — see the module docstring for the rationale.
LIVE_SUPPORTED_PLATFORMS = frozenset({"aws_emr", "aws_glue", "azure_synapse"})


def is_live_pricing_enabled() -> bool:
    """Check whether live pricing is opted in via the environment.

    Returns:
        True when ``SPARK_OPTIMA_LIVE_PRICING`` is set to a truthy value
        (``1``/``true``/``yes``/``on``, case-insensitive), False otherwise.

    """
    return os.environ.get(LIVE_PRICING_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


class PricingCache:
    """Corruption-tolerant JSON file cache for live pricing lookups.

    The cache is a single JSON object mapping the compound key
    ``"<platform>|<region>|<instance_type or ''>"`` to an entry of the form
    ``{"rate": <float or null>, "fetched_at": <unix epoch seconds>}``.
    A ``null`` rate is a *negative* entry recording a failed lookup; negative
    entries expire after :data:`NEGATIVE_TTL_HOURS` regardless of the
    configured positive TTL.

    Any unreadable or malformed cache file is treated as empty (and is
    overwritten on the next :meth:`put`), and write failures are swallowed —
    caching is best-effort and must never break cost estimation.

    Attributes:
        path: Resolved cache file path.
        ttl_hours: Time-to-live for positive entries, in hours.

    """

    def __init__(self, path: Path | None = None, ttl_hours: float | None = None) -> None:
        """Initialize the cache, resolving defaults from the environment.

        Args:
            path: Cache file path. Falls back to the
                ``SPARK_OPTIMA_PRICING_CACHE`` environment variable, then to
                ``~/.spark_optima/pricing_cache.json``.
            ttl_hours: TTL for positive entries in hours. Falls back to the
                ``SPARK_OPTIMA_PRICING_TTL_HOURS`` environment variable, then
                to 24. Invalid environment values fall back to the default.

        """
        if path is None:
            env_path = os.environ.get(CACHE_PATH_ENV_VAR, "").strip()
            path = Path(env_path) if env_path else DEFAULT_CACHE_PATH
        self.path = path

        if ttl_hours is None:
            env_ttl = os.environ.get(CACHE_TTL_ENV_VAR, "").strip()
            ttl_hours = DEFAULT_TTL_HOURS
            if env_ttl:
                try:
                    ttl_hours = float(env_ttl)
                except ValueError:
                    logger.debug(
                        "Invalid %s value %r; using default TTL of %s hours",
                        CACHE_TTL_ENV_VAR,
                        env_ttl,
                        DEFAULT_TTL_HOURS,
                    )
        self.ttl_hours = ttl_hours

    @staticmethod
    def _make_key(platform: str, region: str, instance_type: str | None) -> str:
        """Build the compound cache key for a lookup.

        Args:
            platform: Platform identifier (e.g., "aws_emr").
            region: Cloud region identifier.
            instance_type: Optional instance type (empty for flat rates).

        Returns:
            The compound ``"<platform>|<region>|<instance_type>"`` key.

        """
        return f"{platform.lower()}|{region.lower()}|{(instance_type or '').lower()}"

    def _load(self) -> dict[str, Any]:
        """Load all cache entries, treating any corruption as an empty cache.

        Returns:
            The cached entries, or an empty dict if the file is missing,
            unreadable, not valid JSON, or not a JSON object.

        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def get(self, platform: str, region: str, instance_type: str | None = None) -> tuple[bool, float | None]:
        """Look up a cached rate.

        Args:
            platform: Platform identifier.
            region: Cloud region identifier.
            instance_type: Optional instance type.

        Returns:
            A ``(fresh, rate)`` tuple. ``(True, <float>)`` is a fresh positive
            entry, ``(True, None)`` is a fresh negative entry (a recent failed
            lookup — do not retry yet), and ``(False, None)`` is a miss
            (absent, expired, or malformed entry — fetch again).

        """
        entry = self._load().get(self._make_key(platform, region, instance_type))
        if not isinstance(entry, dict):
            return (False, None)

        fetched_at = entry.get("fetched_at")
        if isinstance(fetched_at, bool) or not isinstance(fetched_at, (int, float)):
            return (False, None)

        age_seconds = time.time() - float(fetched_at)
        if age_seconds < 0:  # Timestamp in the future: clock skew or corruption
            return (False, None)

        rate = entry.get("rate")
        if rate is None:
            # Negative entry: a failed lookup is fresh for a short window only
            if age_seconds <= NEGATIVE_TTL_HOURS * 3600.0:
                return (True, None)
            return (False, None)

        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            return (False, None)
        rate_value = float(rate)
        if not math.isfinite(rate_value) or rate_value <= 0.0:
            return (False, None)

        if age_seconds <= self.ttl_hours * 3600.0:
            return (True, rate_value)
        return (False, None)

    def put(self, platform: str, region: str, instance_type: str | None = None, *, rate: float | None) -> None:
        """Store a lookup result (``rate=None`` records a negative entry).

        Write failures are logged at debug level and otherwise ignored —
        caching is best-effort.

        Args:
            platform: Platform identifier.
            region: Cloud region identifier.
            instance_type: Optional instance type.
            rate: Live USD hourly rate, or None for a failed lookup.

        """
        entries = self._load()
        entries[self._make_key(platform, region, instance_type)] = {
            "rate": rate,
            "fetched_at": time.time(),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not write pricing cache %s: %s", self.path, exc)


class AzureRetailPricesClient:
    """Client for the public Azure Retail Prices API (no authentication).

    The API is documented at
    https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices
    and is queried with an OData ``$filter`` over SKU metadata. For Synapse
    Spark pools the relevant items look like::

        {
            "serviceName": "Azure Synapse Analytics",
            "productName": "Azure Synapse Analytics Serverless Apache Spark Pool - Memory Optimized",
            "meterName": "vCore",
            "skuName": "vCore",
            "armRegionName": "westeurope",
            "priceType": "Consumption",
            "unitOfMeasure": "1 Hour",
            "retailPrice": 0.163,
            ...
        }

    The filter fields are exposed as class attributes so the query is easy to
    adjust if Microsoft renames products or meters.

    Attributes:
        timeout: Request timeout in seconds.

    """

    ENDPOINT = "https://prices.azure.com/api/retail/prices"
    CURRENCY_CODE = "USD"
    # OData $filter building blocks for the Synapse Spark pool vCore SKU.
    # Adjust these if the Azure catalog metadata changes.
    SERVICE_NAME = "Azure Synapse Analytics"
    PRODUCT_NAME_FRAGMENT = "Apache Spark Pool"
    PRICE_TYPE = "Consumption"  # Pay-as-you-go (excludes reservations)
    METER_NAME_FRAGMENT = "vCore"

    def __init__(
        self,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            timeout: Request timeout in seconds (kept short so a slow API
                never stalls cost estimation).
            transport: Optional httpx transport (used by tests to inject a
                ``httpx.MockTransport``).

        """
        self.timeout = timeout
        self._transport = transport

    def build_synapse_vcore_filter(self, region: str) -> str:
        """Build the OData ``$filter`` expression for Synapse vCore pricing.

        Args:
            region: Azure ARM region name (e.g., "eastus", "westeurope").

        Returns:
            The OData filter string for the ``$filter`` query parameter.

        """
        return (
            f"serviceName eq '{self.SERVICE_NAME}'"
            f" and armRegionName eq '{region.lower()}'"
            f" and priceType eq '{self.PRICE_TYPE}'"
            f" and contains(productName, '{self.PRODUCT_NAME_FRAGMENT}')"
        )

    def get_synapse_vcore_rate(self, region: str) -> float:
        """Fetch the live Synapse Spark pool vCore-hour rate for a region.

        Args:
            region: Azure ARM region name (e.g., "eastus").

        Returns:
            The retail USD price per vCore-hour.

        Raises:
            RuntimeError: If httpx is not installed.
            httpx.HTTPError: On connection errors, timeouts, or HTTP error
                status codes.
            LookupError: If the response contains no usable hourly price.
            ValueError: If the response body is not valid JSON.

        """
        if httpx is None:  # pragma: no cover - httpx is normally available
            raise RuntimeError("httpx is required for Azure live pricing. Install with: pip install httpx")

        params = {
            "$filter": self.build_synapse_vcore_filter(region),
            "currencyCode": self.CURRENCY_CODE,
        }
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            response = client.get(self.ENDPOINT, params=params)
            response.raise_for_status()
            payload = response.json()

        items = payload.get("Items") if isinstance(payload, dict) else None
        return self._select_hourly_rate(items if isinstance(items, list) else [])

    def _select_hourly_rate(self, items: list[Any]) -> float:
        """Pick the best matching hourly rate from API result items.

        Preference order: items whose meter name mentions
        :attr:`METER_NAME_FRAGMENT` first, then any positive hourly item.

        Args:
            items: ``Items`` list from the API response.

        Returns:
            The selected USD hourly rate.

        Raises:
            LookupError: If no item has a positive hourly retail price.

        """
        hourly_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            price = item.get("retailPrice")
            if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
                continue
            if "hour" not in str(item.get("unitOfMeasure", "")).lower():
                continue
            hourly_items.append(item)

        for item in hourly_items:
            if self.METER_NAME_FRAGMENT.lower() in str(item.get("meterName", "")).lower():
                return float(item["retailPrice"])
        if hourly_items:
            return float(hourly_items[0]["retailPrice"])

        raise LookupError("No positive hourly retail price found in the Azure Retail Prices response")


class AWSPricingClient:
    """Client for the AWS Pricing API (``GetProducts``) via guarded boto3.

    The AWS Pricing API only exists in a couple of regions; ``us-east-1`` is
    used as the endpoint regardless of the region being priced (the priced
    region goes into the ``regionCode`` filter). boto3 is imported lazily so
    this module imports cleanly without it; a clear ``RuntimeError`` is
    raised only when a lookup is actually attempted without boto3 installed.

    The ``TERM_MATCH`` filter values are exposed as class attributes so the
    queries are easy to adjust if AWS changes its catalog attributes.

    """

    # The Pricing API endpoint region (not the region being priced)
    PRICING_API_REGION = "us-east-1"

    EC2_SERVICE_CODE = "AmazonEC2"
    # Filters selecting the plain on-demand Linux price for an instance type:
    # shared tenancy, no pre-installed software, regular capacity.
    EC2_BASE_FILTERS: dict[str, str] = {
        "operatingSystem": "Linux",
        "tenancy": "Shared",
        "preInstalledSw": "NA",
        "capacitystatus": "Used",
    }

    GLUE_SERVICE_CODE = "AWSGlue"
    # The standard Apache Spark job DPU-hour SKU lives in the
    # "ETL Job run" product group.
    GLUE_BASE_FILTERS: dict[str, str] = {
        "group": "ETL Job run",
    }

    def __init__(self, client: Any | None = None) -> None:
        """Initialize the client.

        Args:
            client: Optional pre-built boto3 ``pricing`` client (used by
                tests). A real client is created lazily when None.

        """
        self._client = client

    def _get_client(self) -> Any:
        """Get (or lazily create) the boto3 ``pricing`` client.

        Returns:
            The boto3 pricing client.

        Raises:
            RuntimeError: If boto3 is not installed.

        """
        if self._client is None:
            try:
                import boto3  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "boto3 is required for AWS live pricing. Install with: pip install boto3",
                ) from e
            self._client = boto3.client("pricing", region_name=self.PRICING_API_REGION)
        return self._client

    @staticmethod
    def _term_match_filters(filters: dict[str, str]) -> list[dict[str, str]]:
        """Convert a field/value mapping to AWS ``TERM_MATCH`` filters.

        Args:
            filters: Mapping of pricing attribute field to exact value.

        Returns:
            Filter dicts in the shape expected by ``GetProducts``.

        """
        return [{"Type": "TERM_MATCH", "Field": field, "Value": value} for field, value in filters.items()]

    @staticmethod
    def _extract_on_demand_usd_rate(price_list: list[Any]) -> float:
        """Extract the first positive on-demand USD rate from a price list.

        ``GetProducts`` returns each product as a JSON *string* (dicts are
        also tolerated). The on-demand hourly rate lives at
        ``terms.OnDemand.<offer>.priceDimensions.<dim>.pricePerUnit.USD``.

        Args:
            price_list: The ``PriceList`` entries from a GetProducts response.

        Returns:
            The first positive USD rate found.

        Raises:
            LookupError: If no positive on-demand USD rate is present.

        """
        for entry in price_list:
            try:
                product = json.loads(entry) if isinstance(entry, str) else entry
            except ValueError:
                continue
            if not isinstance(product, dict):
                continue
            on_demand = product.get("terms", {}).get("OnDemand", {})
            if not isinstance(on_demand, dict):
                continue
            for term in on_demand.values():
                dimensions = term.get("priceDimensions", {}) if isinstance(term, dict) else {}
                if not isinstance(dimensions, dict):
                    continue
                for dimension in dimensions.values():
                    if not isinstance(dimension, dict):
                        continue
                    usd = dimension.get("pricePerUnit", {}).get("USD")
                    try:
                        rate = float(usd)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(rate) and rate > 0:
                        return rate
        raise LookupError("No positive on-demand USD rate found in the AWS Pricing response")

    def get_ec2_ondemand_rate(self, instance_type: str, region: str) -> float:
        """Fetch the live on-demand Linux EC2 hourly rate for an instance type.

        Args:
            instance_type: EC2 instance type (e.g., "m5.xlarge").
            region: AWS region code (e.g., "us-east-1").

        Returns:
            The on-demand USD price per instance-hour (without EMR surcharge).

        Raises:
            RuntimeError: If boto3 is not installed.
            LookupError: If no matching on-demand price is found.

        """
        filters = {"instanceType": instance_type, "regionCode": region, **self.EC2_BASE_FILTERS}
        response = self._get_client().get_products(
            ServiceCode=self.EC2_SERVICE_CODE,
            Filters=self._term_match_filters(filters),
            MaxResults=20,
        )
        return self._extract_on_demand_usd_rate(response.get("PriceList", []))

    def get_glue_dpu_rate(self, region: str) -> float:
        """Fetch the live AWS Glue DPU-hour rate for a region.

        Args:
            region: AWS region code (e.g., "us-east-1").

        Returns:
            The USD price per DPU-hour for Glue ETL job runs.

        Raises:
            RuntimeError: If boto3 is not installed.
            LookupError: If no matching on-demand price is found.

        """
        filters = {"regionCode": region, **self.GLUE_BASE_FILTERS}
        response = self._get_client().get_products(
            ServiceCode=self.GLUE_SERVICE_CODE,
            Filters=self._term_match_filters(filters),
            MaxResults=20,
        )
        return self._extract_on_demand_usd_rate(response.get("PriceList", []))


def _fetch_rate(platform: str, region: str, instance_type: str | None) -> float:
    """Fetch a live rate from the platform's pricing API (may raise).

    Args:
        platform: Normalized (lowercase) platform identifier.
        region: Cloud region identifier.
        instance_type: Instance type (required for aws_emr).

    Returns:
        The live USD hourly rate.

    Raises:
        ValueError: If the platform has no live pricing source.
        Exception: Any client error (network, parse, missing boto3, ...).

    """
    if platform == "azure_synapse":
        return AzureRetailPricesClient().get_synapse_vcore_rate(region)
    if platform == "aws_glue":
        return AWSPricingClient().get_glue_dpu_rate(region)
    if platform == "aws_emr":
        return AWSPricingClient().get_ec2_ondemand_rate(instance_type or "", region)
    raise ValueError(f"No live pricing source for platform '{platform}'")


def get_live_hourly_rate(platform: str, *, region: str, instance_type: str | None = None) -> float | None:
    """Get the live USD hourly rate for a platform, or None on any failure.

    This is the facade used by the platform adapters. It never raises: any
    failure mode (live pricing disabled, unsupported platform, network
    timeout, HTTP error, parse error, missing boto3/credentials, cache
    corruption) results in ``None``, which signals the adapter to fall back
    to its static baseline rate x regional multiplier.

    Platform semantics:

    - ``azure_synapse``: vCore-hour rate (``instance_type`` ignored).
    - ``aws_glue``: DPU-hour rate (``instance_type`` ignored).
    - ``aws_emr``: EC2 on-demand instance-hour rate (``instance_type``
      required; the adapter adds the EMR surcharge on top).
    - ``gcp_dataproc``: always None — the Cloud Billing Catalog API requires
      an API key, so live pricing is deferred (see PLAN.md backlog).
    - ``databricks``: always None — DBU rates are Databricks-proprietary
      list prices with no public pricing API.

    Successful lookups are cached for the configured TTL (24h by default);
    failures are cached as negative entries for one hour.

    Args:
        platform: Platform identifier (e.g., "aws_emr", "azure_synapse").
        region: Cloud region identifier (e.g., "us-east-1", "westeurope").
        instance_type: Instance type for per-instance rates (aws_emr only).

    Returns:
        The live USD hourly rate, or None if live pricing is disabled,
        unsupported for the platform, or the lookup failed.

    """
    try:
        if not is_live_pricing_enabled():
            return None

        normalized = platform.lower()
        if normalized not in LIVE_SUPPORTED_PLATFORMS:
            logger.debug("Live pricing not supported for platform '%s'; using static pricing", platform)
            return None
        if normalized == "aws_emr" and not instance_type:
            logger.debug("Live pricing for aws_emr requires an instance type; using static pricing")
            return None

        cache = PricingCache()
        fresh, cached_rate = cache.get(normalized, region, instance_type)
        if fresh:
            return cached_rate

        rate: float | None
        try:
            rate = _fetch_rate(normalized, region, instance_type)
        except Exception as exc:  # noqa: BLE001 - any client failure means "no live rate"
            logger.debug(
                "Live pricing lookup failed for %s/%s/%s: %s",
                normalized,
                region,
                instance_type or "-",
                exc,
            )
            rate = None

        # Reject nonsensical rates so a bad API response can never produce
        # a zero/negative/NaN cost estimate
        if rate is not None and (
            isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
        ):
            logger.debug("Discarding invalid live rate %r for %s/%s", rate, normalized, region)
            rate = None

        cache.put(normalized, region, instance_type, rate=None if rate is None else float(rate))
        return None if rate is None else float(rate)
    except Exception as exc:  # noqa: BLE001 - the facade must never raise
        logger.debug("Live pricing facade failed for %s/%s: %s", platform, region, exc)
        return None
