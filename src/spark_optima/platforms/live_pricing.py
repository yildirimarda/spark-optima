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
| ``gcp_dataproc`` | GCP Cloud Billing Catalog API (API key)     | N2 machine/h    |

``gcp_dataproc`` is additionally gated on the ``SPARK_OPTIMA_GCP_API_KEY``
environment variable because the Cloud Billing Catalog API cannot be
queried anonymously; without a key the GCP lookup is skipped and static
pricing is used.

``databricks`` deliberately stays on static pricing: DBU rates are
proprietary list prices with no public pricing API (and they vary by
workspace tier and contract), so there is nothing to query.

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
# API key for the GCP Cloud Billing Catalog API (gcp_dataproc only): the
# catalog cannot be queried anonymously, so GCP live pricing is skipped
# entirely when this variable is unset.
GCP_API_KEY_ENV_VAR = "SPARK_OPTIMA_GCP_API_KEY"

# Defaults
DEFAULT_CACHE_PATH = Path.home() / ".spark_optima" / "pricing_cache.json"
DEFAULT_TTL_HOURS = 24.0
# Failed lookups are remembered for a short window so an unreachable pricing
# API does not add a network timeout to every single cost estimate.
NEGATIVE_TTL_HOURS = 1.0
# Live pricing must never noticeably slow down cost estimation
REQUEST_TIMEOUT_SECONDS = 5.0

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

# Platforms with a working live pricing source. databricks is intentionally
# absent — see the module docstring for the rationale. gcp_dataproc is
# additionally gated on SPARK_OPTIMA_GCP_API_KEY at lookup time.
LIVE_SUPPORTED_PLATFORMS = frozenset({"aws_emr", "aws_glue", "azure_synapse", "gcp_dataproc"})


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


class GCPCloudBillingClient:
    """Client for the GCP Cloud Billing Catalog API (API-key gated).

    The Cloud Billing Catalog API
    (https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list)
    lists the public SKUs of one billing service per request. Compute
    Engine's fixed service id in the catalog is ``6F81-5844-456A``, so N2
    machine pricing is read from::

        GET https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus

    The API requires an API key, read from the ``SPARK_OPTIMA_GCP_API_KEY``
    environment variable (or passed explicitly) and sent via the
    ``X-Goog-Api-Key`` request header — never as a URL query parameter — so
    it cannot leak into logged request URLs or ``httpx`` error messages.
    Without a key the client only raises when a lookup is actually
    attempted, so the module imports and constructs cleanly in keyless
    environments.

    N2 machines are priced per vCPU-hour ("core") and per GB-hour ("RAM")
    rather than per machine type. The two SKUs are identified
    **best-effort**: the human-readable description (for example
    ``"N2 Instance Core running in Virginia"``) is matched by prefix, while
    the actual region scoping uses ``serviceRegions`` (region ids such as
    ``us-central1``) and on-demand pricing is selected via
    ``category.usageType == "OnDemand"``. The matching knobs are versioned
    as class attributes so they are easy to adjust if Google rewords the
    catalog. The hourly machine price is then::

        vcpus * core_rate + memory_gb * ram_rate

    Unit prices arrive as ``units`` (whole USD, serialized as a string)
    plus ``nanos`` (billionths of a USD) under
    ``pricingInfo[0].pricingExpression.tieredRates[0].unitPrice``.

    Attributes:
        api_key: Resolved API key, or None when unavailable.
        timeout: Request timeout in seconds.

    """

    ENDPOINT_TEMPLATE = "https://cloudbilling.googleapis.com/v1/services/{service_id}/skus"
    # Compute Engine's fixed service id in the Cloud Billing catalog
    COMPUTE_ENGINE_SERVICE_ID = "6F81-5844-456A"
    CURRENCY_CODE = "USD"
    # Catalog page size (the API caps pageSize at 5000)
    PAGE_SIZE = 5000
    # Hard cap on catalog pages per lookup so a misbehaving API can never
    # stall cost estimation in an endless pagination loop
    MAX_PAGES = 10
    # Total wall-clock budget for one multi-page lookup. Each page fetch is
    # already bounded by the request timeout, but MAX_PAGES sequential slow
    # pages could otherwise stall a cost estimate for ~50 seconds. The
    # deadline is checked between page fetches; when exceeded the lookup
    # gives up (None), which the facade negative-caches for one hour.
    LOOKUP_DEADLINE_SECONDS = 10.0
    # The API key rides in this request header — never in the URL query
    # string — so it cannot leak into logged request URLs or
    # httpx.HTTPStatusError messages.
    API_KEY_HEADER = "X-Goog-Api-Key"
    # Best-effort SKU matching knobs — adjust here if Google rewords the
    # catalog. Descriptions look like "N2 Instance Core running in <region
    # name>"; the trailing human-readable region name is informational only
    # (region filtering uses serviceRegions instead).
    N2_CORE_DESCRIPTION_PREFIX = "N2 Instance Core running in"
    N2_RAM_DESCRIPTION_PREFIX = "N2 Instance Ram running in"
    # Excludes Preemptible/Commit1Yr/... SKUs (compared case-insensitively)
    USAGE_TYPE = "OnDemand"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: Cloud Billing Catalog API key. Falls back to the
                ``SPARK_OPTIMA_GCP_API_KEY`` environment variable; stays
                None (client unusable) when neither is set.
            timeout: Request timeout in seconds (kept short so a slow API
                never stalls cost estimation).
            transport: Optional httpx transport (used by tests to inject a
                ``httpx.MockTransport``).

        """
        if api_key is None:
            api_key = os.environ.get(GCP_API_KEY_ENV_VAR, "").strip() or None
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport

    def get_n2_machine_rate(self, region: str, *, vcpus: float, memory_gb: float) -> float | None:
        """Fetch the live on-demand hourly price of an N2 machine shape.

        Args:
            region: GCP region id (e.g., "us-central1").
            vcpus: Number of vCPUs of the machine type.
            memory_gb: RAM of the machine type in GB.

        Returns:
            The USD hourly machine price: ``vcpus * core_rate + memory_gb *
            ram_rate``, or None when the lookup gave up because the
            :attr:`LOOKUP_DEADLINE_SECONDS` wall-clock deadline expired.

        Raises:
            RuntimeError: If httpx is not installed or no API key is set.
            httpx.HTTPError: On connection errors, timeouts, or HTTP error
                status codes.
            LookupError: If the core or RAM SKU cannot be found within the
                page cap.
            ValueError: If a response body is not valid JSON.

        """
        rates = self.get_n2_core_ram_rates(region)
        if rates is None:
            return None
        core_rate, ram_rate = rates
        return vcpus * core_rate + memory_gb * ram_rate

    def get_n2_core_ram_rates(self, region: str) -> tuple[float, float] | None:
        """Fetch the on-demand N2 core and RAM unit rates for a region.

        Pages through the Compute Engine SKU catalog (following
        ``nextPageToken``, capped at :attr:`MAX_PAGES` pages) and picks the
        first usable core and RAM SKUs matching the region. The wall-clock
        time of the whole lookup is additionally capped at
        :attr:`LOOKUP_DEADLINE_SECONDS` (checked between page fetches).

        Args:
            region: GCP region id (e.g., "us-central1").

        Returns:
            A ``(core_rate, ram_rate)`` tuple in USD per vCPU-hour and USD
            per GB-hour, or None when the per-lookup deadline expired before
            both SKUs were found.

        Raises:
            RuntimeError: If httpx is not installed or no API key is set.
            httpx.HTTPError: On connection errors, timeouts, or HTTP error
                status codes.
            LookupError: If either SKU is missing within the page cap.
            ValueError: If a response body is not valid JSON.

        """
        if httpx is None:  # pragma: no cover - httpx is normally available
            raise RuntimeError("httpx is required for GCP live pricing. Install with: pip install httpx")
        if not self.api_key:
            raise RuntimeError(
                f"A GCP API key is required for the Cloud Billing Catalog API; set {GCP_API_KEY_ENV_VAR}",
            )

        region_lower = region.lower()
        url = self.ENDPOINT_TEMPLATE.format(service_id=self.COMPUTE_ENGINE_SERVICE_ID)
        # The key goes into a header, never into params: httpx logs full
        # request URLs and embeds them in HTTPStatusError messages, so a
        # query-string key would leak into logs.
        headers = {self.API_KEY_HEADER: self.api_key}
        core_rate: float | None = None
        ram_rate: float | None = None
        deadline = time.monotonic() + self.LOOKUP_DEADLINE_SECONDS

        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            page_token: str | None = None
            for _ in range(self.MAX_PAGES):
                params: dict[str, Any] = {
                    "currencyCode": self.CURRENCY_CODE,
                    "pageSize": self.PAGE_SIZE,
                }
                if page_token:
                    params["pageToken"] = page_token
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()

                skus = payload.get("skus") if isinstance(payload, dict) else None
                for sku in skus if isinstance(skus, list) else []:
                    if not self._sku_matches(sku, region_lower):
                        continue
                    description = str(sku.get("description", ""))
                    if core_rate is None and description.startswith(self.N2_CORE_DESCRIPTION_PREFIX):
                        core_rate = self._extract_unit_rate(sku)
                    elif ram_rate is None and description.startswith(self.N2_RAM_DESCRIPTION_PREFIX):
                        ram_rate = self._extract_unit_rate(sku)

                if core_rate is not None and ram_rate is not None:
                    return (core_rate, ram_rate)

                page_token = str(payload.get("nextPageToken") or "") if isinstance(payload, dict) else ""
                if not page_token:
                    break
                if time.monotonic() >= deadline:
                    logger.debug(
                        "GCP pricing lookup for region '%s' exceeded the %.0fs deadline; giving up",
                        region,
                        self.LOOKUP_DEADLINE_SECONDS,
                    )
                    return None

        raise LookupError(
            f"No usable on-demand N2 {'core' if core_rate is None else 'RAM'} SKU found "
            f"for region '{region}' in the Cloud Billing catalog",
        )

    def _sku_matches(self, sku: Any, region_lower: str) -> bool:
        """Check whether a SKU is an on-demand SKU serving the given region.

        Args:
            sku: One entry of the ``skus`` list from the API response.
            region_lower: Lowercased GCP region id to match against
                ``serviceRegions``.

        Returns:
            True when the SKU is a dict with ``category.usageType`` equal to
            :attr:`USAGE_TYPE` (case-insensitive) and ``serviceRegions``
            containing the region.

        """
        if not isinstance(sku, dict):
            return False
        category = sku.get("category")
        usage_type = str(category.get("usageType", "")) if isinstance(category, dict) else ""
        if usage_type.lower() != self.USAGE_TYPE.lower():
            return False
        regions = sku.get("serviceRegions")
        if not isinstance(regions, list):
            return False
        return any(isinstance(entry, str) and entry.lower() == region_lower for entry in regions)

    @staticmethod
    def _extract_unit_rate(sku: dict[str, Any]) -> float | None:
        """Extract the USD unit rate from a SKU's first pricing tier.

        The rate lives at
        ``pricingInfo[0].pricingExpression.tieredRates[0].unitPrice`` as a
        ``units`` (whole USD, serialized as a string) + ``nanos``
        (billionths of a USD) pair.

        Args:
            sku: SKU dict from the API response.

        Returns:
            The positive finite USD rate, or None when the SKU is malformed
            or carries a non-positive price (so a later SKU can still match).

        """
        pricing_info = sku.get("pricingInfo")
        if not isinstance(pricing_info, list) or not pricing_info:
            return None
        first_info = pricing_info[0]
        expression = first_info.get("pricingExpression") if isinstance(first_info, dict) else None
        tiered_rates = expression.get("tieredRates") if isinstance(expression, dict) else None
        if not isinstance(tiered_rates, list) or not tiered_rates:
            return None
        first_tier = tiered_rates[0]
        unit_price = first_tier.get("unitPrice") if isinstance(first_tier, dict) else None
        if not isinstance(unit_price, dict):
            return None

        units = unit_price.get("units", 0)
        nanos = unit_price.get("nanos", 0)
        if isinstance(units, bool) or isinstance(nanos, bool):
            return None
        try:
            rate = float(units or 0) + float(nanos or 0) / 1e9
        except (TypeError, ValueError):
            return None
        if not math.isfinite(rate) or rate <= 0.0:
            return None
        return rate


def _fetch_rate(
    platform: str,
    region: str,
    instance_type: str | None,
    vcpus: float | None = None,
    memory_gb: float | None = None,
) -> float | None:
    """Fetch a live rate from the platform's pricing API (may raise).

    Args:
        platform: Normalized (lowercase) platform identifier.
        region: Cloud region identifier.
        instance_type: Instance type (required for aws_emr and gcp_dataproc).
        vcpus: Machine vCPU count (required for gcp_dataproc).
        memory_gb: Machine RAM in GB (required for gcp_dataproc).

    Returns:
        The live USD hourly rate, or None when the lookup gave up early
        (the GCP per-lookup deadline expired). A None is negative-cached by
        the facade like any other failed lookup.

    Raises:
        ValueError: If the platform has no live pricing source.
        Exception: Any client error (network, parse, missing boto3/API
            key, ...).

    """
    if platform == "azure_synapse":
        return AzureRetailPricesClient().get_synapse_vcore_rate(region)
    if platform == "aws_glue":
        return AWSPricingClient().get_glue_dpu_rate(region)
    if platform == "aws_emr":
        return AWSPricingClient().get_ec2_ondemand_rate(instance_type or "", region)
    if platform == "gcp_dataproc":
        return GCPCloudBillingClient().get_n2_machine_rate(
            region,
            vcpus=float(vcpus or 0.0),
            memory_gb=float(memory_gb or 0.0),
        )
    raise ValueError(f"No live pricing source for platform '{platform}'")


def get_live_hourly_rate(
    platform: str,
    *,
    region: str,
    instance_type: str | None = None,
    vcpus: float | None = None,
    memory_gb: float | None = None,
) -> float | None:
    """Get the live USD hourly rate for a platform, or None on any failure.

    This is the facade used by the platform adapters. It never raises: any
    failure mode (live pricing disabled, unsupported platform, network
    timeout, HTTP error, parse error, missing boto3/credentials, missing
    GCP API key, cache corruption) results in ``None``, which signals the
    adapter to fall back to its static baseline rate x regional multiplier.

    Platform semantics:

    - ``azure_synapse``: vCore-hour rate (``instance_type`` ignored).
    - ``aws_glue``: DPU-hour rate (``instance_type`` ignored).
    - ``aws_emr``: EC2 on-demand instance-hour rate (``instance_type``
      required; the adapter adds the EMR surcharge on top).
    - ``gcp_dataproc``: N2 machine compute-hour rate derived from the
      region's per-core and per-GB SKU rates (``instance_type``, ``vcpus``,
      and ``memory_gb`` required; additionally gated on the
      ``SPARK_OPTIMA_GCP_API_KEY`` environment variable — None when unset.
      The adapter adds the Dataproc fee on top).
    - ``databricks``: always None — DBU rates are Databricks-proprietary
      list prices with no public pricing API.

    Successful lookups are cached for the configured TTL (24h by default);
    failures are cached as negative entries for one hour.

    Args:
        platform: Platform identifier (e.g., "aws_emr", "azure_synapse").
        region: Cloud region identifier (e.g., "us-east-1", "westeurope").
        instance_type: Instance/machine type for per-instance rates
            (aws_emr and gcp_dataproc).
        vcpus: Machine vCPU count (gcp_dataproc only).
        memory_gb: Machine RAM in GB (gcp_dataproc only).

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
        if normalized == "gcp_dataproc":
            if not instance_type or not vcpus or vcpus <= 0 or not memory_gb or memory_gb <= 0:
                logger.debug(
                    "Live pricing for gcp_dataproc requires instance_type, vcpus, and memory_gb; using static pricing",
                )
                return None
            if not os.environ.get(GCP_API_KEY_ENV_VAR, "").strip():
                logger.debug(
                    "Live pricing for gcp_dataproc requires the %s environment variable; using static pricing",
                    GCP_API_KEY_ENV_VAR,
                )
                return None

        cache = PricingCache()
        fresh, cached_rate = cache.get(normalized, region, instance_type)
        if fresh:
            return cached_rate

        rate: float | None
        try:
            rate = _fetch_rate(normalized, region, instance_type, vcpus, memory_gb)
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
