# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the opt-in live pricing layer.

This module contains tests for the Azure Retail Prices client, the guarded
boto3 AWS Pricing client, the API-key-gated GCP Cloud Billing Catalog
client, the JSON file cache, and the never-raising ``get_live_hourly_rate``
facade. All HTTP traffic is mocked with ``httpx.MockTransport`` and boto3 is
replaced with mocks — no test performs real network calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import httpx
import pytest

from spark_optima.platforms import live_pricing
from spark_optima.platforms.live_pricing import (
    DEFAULT_TTL_HOURS,
    GCP_API_KEY_ENV_VAR,
    AWSPricingClient,
    AzureRetailPricesClient,
    GCPCloudBillingClient,
    PricingCache,
    get_live_hourly_rate,
    is_live_pricing_enabled,
)

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Fixtures and helpers
# =============================================================================


@pytest.fixture
def cache_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the pricing cache at a temp file and reset TTL overrides."""
    path = tmp_path / "pricing_cache.json"
    monkeypatch.setenv("SPARK_OPTIMA_PRICING_CACHE", str(path))
    monkeypatch.delenv("SPARK_OPTIMA_PRICING_TTL_HOURS", raising=False)
    return path


@pytest.fixture
def live_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt in to live pricing for the duration of a test."""
    monkeypatch.setenv("SPARK_OPTIMA_LIVE_PRICING", "1")


def _write_cache_entry(
    path: Path,
    key: str,
    rate: float | None,
    age_seconds: float,
) -> None:
    """Write a single cache entry with a back-dated timestamp."""
    path.write_text(
        json.dumps({key: {"rate": rate, "fetched_at": time.time() - age_seconds}}),
        encoding="utf-8",
    )


def _azure_item(
    retail_price: float,
    meter_name: str = "vCore",
    unit_of_measure: str = "1 Hour",
) -> dict[str, Any]:
    """Build a minimal Azure Retail Prices API result item."""
    return {
        "serviceName": "Azure Synapse Analytics",
        "productName": "Azure Synapse Analytics Serverless Apache Spark Pool - Memory Optimized",
        "meterName": meter_name,
        "armRegionName": "westeurope",
        "priceType": "Consumption",
        "unitOfMeasure": unit_of_measure,
        "retailPrice": retail_price,
    }


def _azure_client_with_items(items: list[dict[str, Any]]) -> AzureRetailPricesClient:
    """Build an Azure client whose transport returns the given items."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"Items": items})

    return AzureRetailPricesClient(transport=httpx.MockTransport(handler))


def _ec2_price_list_entry(usd: str) -> str:
    """Build a GetProducts PriceList entry (JSON string) with one rate."""
    return json.dumps(
        {
            "product": {"attributes": {"instanceType": "m5.xlarge"}},
            "terms": {
                "OnDemand": {
                    "OFFER.CODE": {
                        "priceDimensions": {
                            "OFFER.CODE.DIM": {
                                "unit": "Hrs",
                                "pricePerUnit": {"USD": usd},
                            },
                        },
                    },
                },
            },
        },
    )


def _mock_pricing_client(price_list: list[Any]) -> MagicMock:
    """Build a fake boto3 pricing client returning the given PriceList."""
    client = MagicMock()
    client.get_products.return_value = {"PriceList": price_list}
    return client


def _gcp_sku(
    description: str,
    *,
    units: Any = "0",
    nanos: Any = 0,
    usage_type: str = "OnDemand",
    regions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal Cloud Billing Catalog SKU entry."""
    return {
        "skuId": "0000-0000-0000",
        "description": description,
        "category": {
            "resourceFamily": "Compute",
            "resourceGroup": "N2Standard",
            "usageType": usage_type,
        },
        "serviceRegions": regions if regions is not None else ["us-central1"],
        "pricingInfo": [
            {
                "pricingExpression": {
                    "usageUnit": "h",
                    "tieredRates": [
                        {
                            "startUsageAmount": 0,
                            "unitPrice": {"currencyCode": "USD", "units": units, "nanos": nanos},
                        },
                    ],
                },
            },
        ],
    }


def _gcp_core_sku(nanos: int = 31611000, **kwargs: Any) -> dict[str, Any]:
    """Build an N2 core SKU (default $0.031611 per vCPU-hour)."""
    return _gcp_sku("N2 Instance Core running in Iowa", nanos=nanos, **kwargs)


def _gcp_ram_sku(nanos: int = 4237000, **kwargs: Any) -> dict[str, Any]:
    """Build an N2 RAM SKU (default $0.004237 per GB-hour)."""
    return _gcp_sku("N2 Instance Ram running in Iowa", nanos=nanos, **kwargs)


def _gcp_transport_for_pages(
    pages: list[list[dict[str, Any]]],
    seen: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """Serve canned SKU pages chained via numeric ``nextPageToken`` values."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        token = request.url.params.get("pageToken", "")
        index = int(token) if token else 0
        body: dict[str, Any] = {"skus": pages[min(index, len(pages) - 1)]}
        if index < len(pages) - 1:
            body["nextPageToken"] = str(index + 1)
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _gcp_client_with_pages(
    pages: list[list[dict[str, Any]]],
    seen: list[httpx.Request] | None = None,
) -> GCPCloudBillingClient:
    """Build a GCP client whose transport serves the given SKU pages."""
    return GCPCloudBillingClient(api_key="test-key", transport=_gcp_transport_for_pages(pages, seen))


# =============================================================================
# is_live_pricing_enabled
# =============================================================================


class TestIsLivePricingEnabled:
    """Test cases for the SPARK_OPTIMA_LIVE_PRICING opt-in flag."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " 1 "])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """Test that truthy values enable live pricing."""
        monkeypatch.setenv("SPARK_OPTIMA_LIVE_PRICING", value)
        assert is_live_pricing_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "", "no", "off", "enable"])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """Test that falsy or unknown values keep live pricing disabled."""
        monkeypatch.setenv("SPARK_OPTIMA_LIVE_PRICING", value)
        assert is_live_pricing_enabled() is False

    def test_unset_is_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the default (unset) state is disabled."""
        monkeypatch.delenv("SPARK_OPTIMA_LIVE_PRICING", raising=False)
        assert is_live_pricing_enabled() is False


# =============================================================================
# PricingCache
# =============================================================================


class TestPricingCache:
    """Test cases for the JSON file pricing cache."""

    def test_put_get_roundtrip(self, cache_path: Path) -> None:
        """Test that a stored positive rate is returned fresh."""
        cache = PricingCache()
        cache.put("aws_glue", "us-east-1", rate=0.44)

        assert cache.get("aws_glue", "us-east-1") == (True, 0.44)
        assert cache_path.exists()

    def test_get_missing_file_is_miss(self, cache_path: Path) -> None:
        """Test that a missing cache file yields a miss."""
        assert PricingCache().get("aws_glue", "us-east-1") == (False, None)

    def test_keys_include_instance_type(self, cache_path: Path) -> None:
        """Test that entries are keyed by (platform, region, instance_type)."""
        cache = PricingCache()
        cache.put("aws_emr", "us-east-1", "m5.xlarge", rate=0.192)
        cache.put("aws_emr", "us-east-1", "r5.xlarge", rate=0.252)

        assert cache.get("aws_emr", "us-east-1", "m5.xlarge") == (True, 0.192)
        assert cache.get("aws_emr", "us-east-1", "r5.xlarge") == (True, 0.252)
        assert cache.get("aws_emr", "us-east-1") == (False, None)

    def test_keys_are_case_insensitive(self, cache_path: Path) -> None:
        """Test that lookups normalize key casing."""
        cache = PricingCache()
        cache.put("AWS_EMR", "US-EAST-1", "M5.XLARGE", rate=0.192)

        assert cache.get("aws_emr", "us-east-1", "m5.xlarge") == (True, 0.192)

    def test_positive_entry_expires_after_ttl(self, cache_path: Path) -> None:
        """Test that a positive entry older than the TTL is a miss."""
        _write_cache_entry(cache_path, "aws_glue|us-east-1|", rate=0.44, age_seconds=25 * 3600)

        assert PricingCache().get("aws_glue", "us-east-1") == (False, None)

    def test_positive_entry_fresh_within_ttl(self, cache_path: Path) -> None:
        """Test that a positive entry younger than the TTL is fresh."""
        _write_cache_entry(cache_path, "aws_glue|us-east-1|", rate=0.44, age_seconds=23 * 3600)

        assert PricingCache().get("aws_glue", "us-east-1") == (True, 0.44)

    def test_negative_entry_fresh_within_one_hour(self, cache_path: Path) -> None:
        """Test that a recent failed lookup is returned as a fresh negative."""
        _write_cache_entry(cache_path, "aws_glue|us-east-1|", rate=None, age_seconds=0.5 * 3600)

        assert PricingCache().get("aws_glue", "us-east-1") == (True, None)

    def test_negative_entry_expires_after_one_hour(self, cache_path: Path) -> None:
        """Test that a stale failed lookup becomes a miss (retry allowed)."""
        _write_cache_entry(cache_path, "aws_glue|us-east-1|", rate=None, age_seconds=2 * 3600)

        assert PricingCache().get("aws_glue", "us-east-1") == (False, None)

    def test_ttl_env_override(self, cache_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that SPARK_OPTIMA_PRICING_TTL_HOURS shortens the TTL."""
        monkeypatch.setenv("SPARK_OPTIMA_PRICING_TTL_HOURS", "1")
        _write_cache_entry(cache_path, "aws_glue|us-east-1|", rate=0.44, age_seconds=2 * 3600)

        assert PricingCache().get("aws_glue", "us-east-1") == (False, None)

    def test_invalid_ttl_env_falls_back_to_default(self, cache_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that an unparsable TTL value falls back to the default."""
        monkeypatch.setenv("SPARK_OPTIMA_PRICING_TTL_HOURS", "soon")

        assert PricingCache().ttl_hours == DEFAULT_TTL_HOURS

    def test_cache_path_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that SPARK_OPTIMA_PRICING_CACHE relocates the cache file."""
        custom = tmp_path / "nested" / "custom_cache.json"
        monkeypatch.setenv("SPARK_OPTIMA_PRICING_CACHE", str(custom))

        cache = PricingCache()
        cache.put("aws_glue", "us-east-1", rate=0.44)

        assert cache.path == custom
        assert custom.exists()

    def test_corrupt_json_is_treated_as_empty(self, cache_path: Path) -> None:
        """Test that invalid JSON yields a miss instead of raising."""
        cache_path.write_text("{not valid json", encoding="utf-8")

        assert PricingCache().get("aws_glue", "us-east-1") == (False, None)

    def test_corrupt_json_is_overwritten_on_put(self, cache_path: Path) -> None:
        """Test that put recovers from a corrupt cache file."""
        cache_path.write_text("\x00garbage", encoding="utf-8")

        cache = PricingCache()
        cache.put("aws_glue", "us-east-1", rate=0.44)

        assert cache.get("aws_glue", "us-east-1") == (True, 0.44)
        assert isinstance(json.loads(cache_path.read_text(encoding="utf-8")), dict)

    def test_non_object_json_root_is_miss(self, cache_path: Path) -> None:
        """Test that a JSON root that is not an object is treated as empty."""
        cache_path.write_text("[1, 2, 3]", encoding="utf-8")

        assert PricingCache().get("aws_glue", "us-east-1") == (False, None)

    @pytest.mark.parametrize(
        "entry",
        [
            "not-a-dict",
            {"rate": 0.44},  # missing fetched_at
            {"rate": 0.44, "fetched_at": "yesterday"},  # bad timestamp type
            {"rate": "0.44", "fetched_at": 0},  # bad rate type (also stale)
            {"rate": True, "fetched_at": 0},  # bool is not a usable rate
            {"rate": -1.0, "fetched_at": 0},  # non-positive rate
        ],
    )
    def test_malformed_entries_are_misses(self, cache_path: Path, entry: Any) -> None:
        """Test that malformed entries yield a miss instead of raising."""
        cache_path.write_text(json.dumps({"aws_glue|us-east-1|": entry}), encoding="utf-8")

        assert PricingCache().get("aws_glue", "us-east-1") == (False, None)

    def test_future_timestamp_is_miss(self, cache_path: Path) -> None:
        """Test that a timestamp in the future (clock skew) is a miss."""
        _write_cache_entry(cache_path, "aws_glue|us-east-1|", rate=0.44, age_seconds=-3600)

        assert PricingCache().get("aws_glue", "us-east-1") == (False, None)

    def test_put_failure_is_swallowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that an unwritable cache path does not raise."""
        blocker = tmp_path / "blocker"
        blocker.write_text("file, not a directory", encoding="utf-8")
        monkeypatch.setenv("SPARK_OPTIMA_PRICING_CACHE", str(blocker / "cache.json"))

        PricingCache().put("aws_glue", "us-east-1", rate=0.44)  # Must not raise


# =============================================================================
# AzureRetailPricesClient
# =============================================================================


class TestAzureRetailPricesClient:
    """Test cases for the Azure Retail Prices API client (httpx mocked)."""

    def test_get_synapse_vcore_rate_success(self) -> None:
        """Test that the vCore retail price is returned."""
        client = _azure_client_with_items([_azure_item(0.163)])

        assert client.get_synapse_vcore_rate("westeurope") == 0.163

    def test_request_url_and_filter(self) -> None:
        """Test the query construction: endpoint, $filter fields, currency."""
        seen: dict[str, httpx.Request] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            return httpx.Response(200, json={"Items": [_azure_item(0.163)]})

        client = AzureRetailPricesClient(transport=httpx.MockTransport(handler))
        client.get_synapse_vcore_rate("WestEurope")

        request = seen["request"]
        assert request.url.host == "prices.azure.com"
        assert request.url.path == "/api/retail/prices"
        params = dict(request.url.params)
        assert params["currencyCode"] == "USD"
        odata_filter = params["$filter"]
        assert "serviceName eq 'Azure Synapse Analytics'" in odata_filter
        assert "armRegionName eq 'westeurope'" in odata_filter  # Normalized to lowercase
        assert "priceType eq 'Consumption'" in odata_filter
        assert "contains(productName, 'Apache Spark Pool')" in odata_filter

    def test_prefers_vcore_meter(self) -> None:
        """Test that a vCore meter wins over other hourly meters."""
        items = [
            _azure_item(9.99, meter_name="Some Other Meter"),
            _azure_item(0.163, meter_name="vCore"),
        ]

        assert _azure_client_with_items(items).get_synapse_vcore_rate("eastus") == 0.163

    def test_falls_back_to_first_positive_hourly_item(self) -> None:
        """Test the fallback when no meter name mentions vCore."""
        items = [
            _azure_item(0.0, meter_name="Free Tier"),  # Skipped: zero price
            _azure_item(0.5, meter_name="Spark Pool Compute", unit_of_measure="1/Month"),  # Skipped: not hourly
            _azure_item(0.25, meter_name="Spark Pool Compute"),
        ]

        assert _azure_client_with_items(items).get_synapse_vcore_rate("eastus") == 0.25

    def test_no_usable_items_raises_lookup_error(self) -> None:
        """Test that an empty result raises LookupError."""
        with pytest.raises(LookupError):
            _azure_client_with_items([]).get_synapse_vcore_rate("eastus")

    def test_http_error_raises(self) -> None:
        """Test that a 5xx response raises an httpx error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server error")

        client = AzureRetailPricesClient(transport=httpx.MockTransport(handler))
        with pytest.raises(httpx.HTTPStatusError):
            client.get_synapse_vcore_rate("eastus")

    def test_timeout_raises(self) -> None:
        """Test that a timeout propagates as an httpx exception."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("connection timed out")

        client = AzureRetailPricesClient(transport=httpx.MockTransport(handler))
        with pytest.raises(httpx.TimeoutException):
            client.get_synapse_vcore_rate("eastus")

    def test_invalid_json_body_raises(self) -> None:
        """Test that a non-JSON body raises a ValueError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        client = AzureRetailPricesClient(transport=httpx.MockTransport(handler))
        with pytest.raises(ValueError):
            client.get_synapse_vcore_rate("eastus")


# =============================================================================
# AWSPricingClient
# =============================================================================


class TestAWSPricingClient:
    """Test cases for the guarded boto3 AWS Pricing client (boto3 mocked)."""

    def test_get_ec2_ondemand_rate_success(self) -> None:
        """Test EC2 rate extraction from a GetProducts response."""
        fake = _mock_pricing_client([_ec2_price_list_entry("0.1920000000")])

        rate = AWSPricingClient(client=fake).get_ec2_ondemand_rate("m5.xlarge", "us-east-1")

        assert rate == pytest.approx(0.192)

    def test_ec2_filters_and_service_code(self) -> None:
        """Test the GetProducts query construction for EC2."""
        fake = _mock_pricing_client([_ec2_price_list_entry("0.214")])

        AWSPricingClient(client=fake).get_ec2_ondemand_rate("m5.xlarge", "eu-west-1")

        kwargs = fake.get_products.call_args.kwargs
        assert kwargs["ServiceCode"] == "AmazonEC2"
        filters = {f["Field"]: f["Value"] for f in kwargs["Filters"]}
        assert all(f["Type"] == "TERM_MATCH" for f in kwargs["Filters"])
        assert filters["instanceType"] == "m5.xlarge"
        assert filters["regionCode"] == "eu-west-1"
        assert filters["operatingSystem"] == "Linux"
        assert filters["tenancy"] == "Shared"
        assert filters["preInstalledSw"] == "NA"
        assert filters["capacitystatus"] == "Used"

    def test_get_glue_dpu_rate_success(self) -> None:
        """Test Glue DPU rate extraction and query construction."""
        fake = _mock_pricing_client([_ec2_price_list_entry("0.44")])

        rate = AWSPricingClient(client=fake).get_glue_dpu_rate("us-east-1")

        assert rate == pytest.approx(0.44)
        kwargs = fake.get_products.call_args.kwargs
        assert kwargs["ServiceCode"] == "AWSGlue"
        filters = {f["Field"]: f["Value"] for f in kwargs["Filters"]}
        assert filters["regionCode"] == "us-east-1"
        assert filters["group"] == "ETL Job run"

    def test_price_list_dict_entries_are_tolerated(self) -> None:
        """Test that already-decoded PriceList entries (dicts) work too."""
        fake = _mock_pricing_client([json.loads(_ec2_price_list_entry("0.34"))])

        rate = AWSPricingClient(client=fake).get_ec2_ondemand_rate("c5.2xlarge", "us-east-1")

        assert rate == pytest.approx(0.34)

    def test_zero_prices_are_skipped(self) -> None:
        """Test that zero-priced dimensions are skipped for the next entry."""
        fake = _mock_pricing_client(
            [_ec2_price_list_entry("0.0000000000"), _ec2_price_list_entry("0.192")],
        )

        rate = AWSPricingClient(client=fake).get_ec2_ondemand_rate("m5.xlarge", "us-east-1")

        assert rate == pytest.approx(0.192)

    def test_malformed_entries_are_skipped(self) -> None:
        """Test that unparsable entries are skipped without raising."""
        fake = _mock_pricing_client(["{broken json", 42, _ec2_price_list_entry("0.192")])

        rate = AWSPricingClient(client=fake).get_ec2_ondemand_rate("m5.xlarge", "us-east-1")

        assert rate == pytest.approx(0.192)

    def test_empty_price_list_raises_lookup_error(self) -> None:
        """Test that an empty PriceList raises LookupError."""
        fake = _mock_pricing_client([])

        with pytest.raises(LookupError):
            AWSPricingClient(client=fake).get_ec2_ondemand_rate("m5.xlarge", "us-east-1")

    def test_missing_boto3_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the boto3-absent path: a clear RuntimeError only on use."""
        # Setting sys.modules["boto3"] to None makes `import boto3` raise
        # ImportError deterministically, even when boto3 is installed
        monkeypatch.setitem(sys.modules, "boto3", None)

        client = AWSPricingClient()  # Construction must succeed without boto3

        with pytest.raises(RuntimeError, match="boto3"):
            client.get_glue_dpu_rate("us-east-1")

    def test_module_importable_without_boto3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the module has no top-level boto3 dependency."""
        monkeypatch.setitem(sys.modules, "boto3", None)

        # Re-executing the module source must not raise ImportError
        spec = importlib.util.spec_from_file_location("_live_pricing_reimport", live_pricing.__file__)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "get_live_hourly_rate")


class TestAWSPricingClientWithRealBoto3:
    """Test cases that need the real boto3 module installed."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("boto3") is None,
        reason="boto3 not installed",
    )

    def test_lazy_client_uses_us_east_1_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the pricing client targets the us-east-1 API endpoint."""
        boto3 = pytest.importorskip("boto3")
        fake = _mock_pricing_client([_ec2_price_list_entry("0.192")])
        factory = MagicMock(return_value=fake)
        monkeypatch.setattr(boto3, "client", factory)

        rate = AWSPricingClient().get_ec2_ondemand_rate("m5.xlarge", "eu-west-1")

        assert rate == pytest.approx(0.192)
        factory.assert_called_once_with("pricing", region_name="us-east-1")


# =============================================================================
# GCPCloudBillingClient
# =============================================================================


class TestGCPCloudBillingClient:
    """Test cases for the GCP Cloud Billing Catalog client (httpx mocked)."""

    def test_get_n2_machine_rate_success(self) -> None:
        """Test the core/RAM SKU combination into a machine-hour price."""
        client = _gcp_client_with_pages([[_gcp_core_sku(), _gcp_ram_sku()]])

        rate = client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)

        # 8 vCPUs x $0.031611 + 32 GB x $0.004237
        assert rate == pytest.approx(8 * 0.031611 + 32 * 0.004237)

    def test_request_url_service_id_and_params(self) -> None:
        """Test the query construction: endpoint, service id, currency."""
        seen: list[httpx.Request] = []
        client = _gcp_client_with_pages([[_gcp_core_sku(), _gcp_ram_sku()]], seen)

        client.get_n2_machine_rate("us-central1", vcpus=4, memory_gb=16)

        request = seen[0]
        assert request.url.host == "cloudbilling.googleapis.com"
        # 6F81-5844-456A is Compute Engine's fixed Cloud Billing service id
        assert request.url.path == "/v1/services/6F81-5844-456A/skus"
        params = dict(request.url.params)
        assert params["currencyCode"] == "USD"
        assert params["pageSize"] == str(GCPCloudBillingClient.PAGE_SIZE)

    def test_api_key_sent_as_header_and_never_in_url(self) -> None:
        """Regression: the key rides in X-Goog-Api-Key, never in the (logged) URL.

        httpx logs full request URLs at INFO level and embeds them in
        HTTPStatusError messages, so a query-string key would leak the
        secret into logs.
        """
        seen: list[httpx.Request] = []
        client = _gcp_client_with_pages([[_gcp_core_sku()], [_gcp_ram_sku()]], seen)

        client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)

        assert len(seen) == 2  # Every page request must carry the header
        for request in seen:
            assert request.headers["X-Goog-Api-Key"] == "test-key"
            assert "key" not in dict(request.url.params)
            assert "test-key" not in str(request.url)

    def test_lookup_deadline_aborts_slow_pagination(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that slow pages abort at the deadline instead of paging to MAX_PAGES.

        The clock is monkeypatched (no real sleeping): each page "takes" 60%
        of the deadline, so page 1 leaves budget for one more fetch and the
        lookup must give up (None) right after page 2.
        """
        requests: list[httpx.Request] = []
        clock = {"now": 0.0}
        page_seconds = GCPCloudBillingClient.LOOKUP_DEADLINE_SECONDS * 0.6

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            clock["now"] += page_seconds  # Simulate a slow page without sleeping
            # Never contains N2 SKUs and always advertises another page
            return httpx.Response(200, json={"skus": [_gcp_sku("Unrelated SKU")], "nextPageToken": "more"})

        monkeypatch.setattr(live_pricing, "time", SimpleNamespace(monotonic=lambda: clock["now"], time=time.time))
        client = GCPCloudBillingClient(api_key="test-key", transport=httpx.MockTransport(handler))

        assert client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32) is None
        assert len(requests) == 2  # Page 1 ends below the deadline, page 2 crosses it
        assert len(requests) < GCPCloudBillingClient.MAX_PAGES

    @pytest.mark.parametrize(
        ("units", "nanos", "expected"),
        [
            ("0", 31611000, 0.031611),  # Sub-dollar rate from nanos only
            ("1", 250000000, 1.25),  # Units + nanos combine
            ("2", 0, 2.0),  # Whole-dollar rate, no nanos
            (3, 500000000, 3.5),  # Numeric units are tolerated too
        ],
    )
    def test_units_and_nanos_to_usd(self, units: Any, nanos: int, expected: float) -> None:
        """Test the units + nanos -> USD float conversion exactly."""
        core = _gcp_sku("N2 Instance Core running in Iowa", units=units, nanos=nanos)
        client = _gcp_client_with_pages([[core, _gcp_ram_sku(nanos=1000000)]])

        rate = client.get_n2_machine_rate("us-central1", vcpus=1, memory_gb=0)

        assert rate == pytest.approx(expected, rel=1e-12)

    def test_multi_page_pagination_follows_next_page_token(self) -> None:
        """Test that the RAM SKU is found on a later page via nextPageToken."""
        seen: list[httpx.Request] = []
        client = _gcp_client_with_pages(
            [
                [_gcp_sku("Unrelated E2 SKU"), _gcp_core_sku()],
                [_gcp_sku("Another unrelated SKU")],
                [_gcp_ram_sku()],
            ],
            seen,
        )

        rate = client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)

        assert rate == pytest.approx(8 * 0.031611 + 32 * 0.004237)
        assert len(seen) == 3
        assert seen[1].url.params["pageToken"] == "1"
        assert seen[2].url.params["pageToken"] == "2"

    def test_pagination_stops_at_hard_page_cap(self) -> None:
        """Test that a never-ending catalog stops at MAX_PAGES requests."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            # Always advertises another page and never contains N2 SKUs
            return httpx.Response(200, json={"skus": [_gcp_sku("Unrelated SKU")], "nextPageToken": "more"})

        client = GCPCloudBillingClient(api_key="test-key", transport=httpx.MockTransport(handler))

        with pytest.raises(LookupError):
            client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)
        assert len(requests) == GCPCloudBillingClient.MAX_PAGES

    def test_stops_fetching_once_both_rates_found(self) -> None:
        """Test that no extra page is fetched after both SKUs matched."""
        seen: list[httpx.Request] = []
        client = _gcp_client_with_pages(
            [[_gcp_core_sku(), _gcp_ram_sku()], [_gcp_sku("Never fetched")]],
            seen,
        )

        client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)

        assert len(seen) == 1

    def test_region_filtering_uses_service_regions(self) -> None:
        """Test that SKUs of other regions are skipped via serviceRegions."""
        skus = [
            _gcp_core_sku(nanos=99999000, regions=["europe-west1"]),  # Wrong region
            _gcp_ram_sku(nanos=88888000, regions=["europe-west1"]),  # Wrong region
            _gcp_core_sku(regions=["us-east1", "us-central1"]),  # Multi-region entry matches
            _gcp_ram_sku(regions=["US-CENTRAL1"]),  # Region match is case-insensitive
        ]

        rate = _gcp_client_with_pages([skus]).get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)

        assert rate == pytest.approx(8 * 0.031611 + 32 * 0.004237)

    def test_non_ondemand_usage_types_are_skipped(self) -> None:
        """Test that preemptible/committed SKUs never win over OnDemand."""
        skus = [
            _gcp_core_sku(nanos=7000000, usage_type="Preemptible"),  # Cheaper but not on-demand
            _gcp_core_sku(nanos=7000000, usage_type="Commit1Yr"),
            _gcp_core_sku(),
            _gcp_ram_sku(),
        ]

        rate = _gcp_client_with_pages([skus]).get_n2_machine_rate("us-central1", vcpus=1, memory_gb=0)

        assert rate == pytest.approx(0.031611)

    @pytest.mark.parametrize(
        "skus",
        [
            [],  # Empty catalog
            [_gcp_core_sku()],  # RAM SKU missing
            [_gcp_ram_sku()],  # Core SKU missing
        ],
    )
    def test_missing_skus_raise_lookup_error(self, skus: list[dict[str, Any]]) -> None:
        """Test that a missing core or RAM SKU raises LookupError."""
        with pytest.raises(LookupError):
            _gcp_client_with_pages([skus]).get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)

    def test_malformed_sku_is_skipped_for_next_match(self) -> None:
        """Test that a matching SKU without usable pricing does not block a later one."""
        broken_core = _gcp_core_sku()
        broken_core["pricingInfo"] = []  # No pricing tiers at all
        zero_priced_ram = _gcp_ram_sku(nanos=0)  # Non-positive rate is rejected
        skus = [broken_core, zero_priced_ram, "not-a-dict", _gcp_core_sku(), _gcp_ram_sku()]

        rate = _gcp_client_with_pages([skus]).get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)  # type: ignore[list-item]

        assert rate == pytest.approx(8 * 0.031611 + 32 * 0.004237)

    def test_no_api_key_raises_runtime_error_only_on_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the keyless path: construction succeeds, lookup raises."""
        monkeypatch.delenv(GCP_API_KEY_ENV_VAR, raising=False)

        client = GCPCloudBillingClient()  # Construction must succeed without a key

        assert client.api_key is None
        with pytest.raises(RuntimeError, match=GCP_API_KEY_ENV_VAR):
            client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)

    def test_api_key_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the API key falls back to SPARK_OPTIMA_GCP_API_KEY."""
        monkeypatch.setenv(GCP_API_KEY_ENV_VAR, "env-key")

        assert GCPCloudBillingClient().api_key == "env-key"

    def test_http_error_raises(self) -> None:
        """Test that a 403 (bad API key) raises an httpx error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": {"message": "API key not valid"}})

        client = GCPCloudBillingClient(api_key="bad-key", transport=httpx.MockTransport(handler))

        with pytest.raises(httpx.HTTPStatusError):
            client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)

    def test_timeout_raises(self) -> None:
        """Test that a timeout propagates as an httpx exception."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("connection timed out")

        client = GCPCloudBillingClient(api_key="test-key", transport=httpx.MockTransport(handler))

        with pytest.raises(httpx.TimeoutException):
            client.get_n2_machine_rate("us-central1", vcpus=8, memory_gb=32)


# =============================================================================
# get_live_hourly_rate facade
# =============================================================================


class TestGetLiveHourlyRateFacade:
    """Test cases for the never-raising live pricing facade."""

    def _patch_azure(self, monkeypatch: pytest.MonkeyPatch, result: Any) -> MagicMock:
        """Replace the Azure client class with a mock returning ``result``."""
        instance = MagicMock()
        if isinstance(result, BaseException):
            instance.get_synapse_vcore_rate.side_effect = result
        else:
            instance.get_synapse_vcore_rate.return_value = result
        monkeypatch.setattr(live_pricing, "AzureRetailPricesClient", MagicMock(return_value=instance))
        return instance

    def _patch_aws(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Replace the AWS client class with a mock and return the instance."""
        instance = MagicMock()
        monkeypatch.setattr(live_pricing, "AWSPricingClient", MagicMock(return_value=instance))
        return instance

    def test_disabled_returns_none_without_side_effects(
        self,
        cache_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that the default-off path is inert (no fetch, no cache file)."""
        monkeypatch.delenv("SPARK_OPTIMA_LIVE_PRICING", raising=False)
        fetch = MagicMock()
        monkeypatch.setattr(live_pricing, "_fetch_rate", fetch)

        assert get_live_hourly_rate("azure_synapse", region="eastus") is None
        fetch.assert_not_called()
        assert not cache_path.exists()

    def test_falsy_env_value_returns_none(self, cache_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a falsy env value keeps live pricing off."""
        monkeypatch.setenv("SPARK_OPTIMA_LIVE_PRICING", "0")

        assert get_live_hourly_rate("aws_glue", region="us-east-1") is None
        assert not cache_path.exists()

    @pytest.mark.parametrize("platform", ["databricks", "local", "unknown"])
    def test_unsupported_platforms_return_none(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
        platform: str,
    ) -> None:
        """Test that platforms without a live source always return None."""
        fetch = MagicMock()
        monkeypatch.setattr(live_pricing, "_fetch_rate", fetch)

        assert get_live_hourly_rate(platform, region="anywhere") is None
        fetch.assert_not_called()

    def test_aws_emr_requires_instance_type(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that aws_emr without an instance type returns None."""
        fetch = MagicMock()
        monkeypatch.setattr(live_pricing, "_fetch_rate", fetch)

        assert get_live_hourly_rate("aws_emr", region="us-east-1") is None
        fetch.assert_not_called()

    def test_success_returns_and_caches_rate(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the happy path: live rate returned and written to the cache."""
        instance = self._patch_azure(monkeypatch, 0.163)

        assert get_live_hourly_rate("azure_synapse", region="westeurope") == 0.163

        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached["azure_synapse|westeurope|"]["rate"] == 0.163
        assert instance.get_synapse_vcore_rate.call_count == 1

    def test_second_call_served_from_cache(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a fresh cache entry avoids a second fetch."""
        instance = self._patch_azure(monkeypatch, 0.163)

        first = get_live_hourly_rate("azure_synapse", region="westeurope")
        second = get_live_hourly_rate("azure_synapse", region="westeurope")

        assert first == second == 0.163
        assert instance.get_synapse_vcore_rate.call_count == 1

    def test_aws_glue_dispatch(self, cache_path: Path, live_enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that aws_glue lookups hit the Glue DPU rate."""
        instance = self._patch_aws(monkeypatch)
        instance.get_glue_dpu_rate.return_value = 0.44

        assert get_live_hourly_rate("aws_glue", region="eu-west-1") == 0.44
        instance.get_glue_dpu_rate.assert_called_once_with("eu-west-1")

    def test_aws_emr_dispatch(self, cache_path: Path, live_enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that aws_emr lookups hit the EC2 on-demand rate."""
        instance = self._patch_aws(monkeypatch)
        instance.get_ec2_ondemand_rate.return_value = 0.192

        rate = get_live_hourly_rate("aws_emr", region="us-east-1", instance_type="m5.xlarge")

        assert rate == 0.192
        instance.get_ec2_ondemand_rate.assert_called_once_with("m5.xlarge", "us-east-1")

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectTimeout("timed out"),
            httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock()),
            LookupError("no items"),
            ValueError("bad json"),
            RuntimeError("boto3 is required"),
            KeyError("Items"),
        ],
    )
    def test_any_fetch_failure_returns_none(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
        error: BaseException,
    ) -> None:
        """Test that every fetch failure mode yields None, never an exception."""
        self._patch_azure(monkeypatch, error)

        assert get_live_hourly_rate("azure_synapse", region="eastus") is None

    def test_failure_is_negative_cached(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a failed lookup is cached and not retried immediately."""
        instance = self._patch_azure(monkeypatch, httpx.ConnectTimeout("timed out"))

        assert get_live_hourly_rate("azure_synapse", region="eastus") is None
        assert get_live_hourly_rate("azure_synapse", region="eastus") is None

        assert instance.get_synapse_vcore_rate.call_count == 1  # Second call hit the negative entry
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached["azure_synapse|eastus|"]["rate"] is None

    def test_expired_negative_entry_triggers_refetch(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that an expired negative entry allows a new fetch."""
        _write_cache_entry(cache_path, "azure_synapse|eastus|", rate=None, age_seconds=2 * 3600)
        instance = self._patch_azure(monkeypatch, 0.151)

        assert get_live_hourly_rate("azure_synapse", region="eastus") == 0.151
        assert instance.get_synapse_vcore_rate.call_count == 1

    @pytest.mark.parametrize("bad_rate", [0.0, -1.0, float("nan"), float("inf"), "0.44", None, True])
    def test_invalid_fetched_rates_return_none(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
        bad_rate: Any,
    ) -> None:
        """Test that nonsensical fetched rates are rejected."""
        self._patch_azure(monkeypatch, bad_rate)

        assert get_live_hourly_rate("azure_synapse", region="eastus") is None

    def test_corrupt_cache_does_not_block_fetch(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a corrupt cache file still allows a live lookup."""
        cache_path.write_text("{broken", encoding="utf-8")
        self._patch_azure(monkeypatch, 0.163)

        assert get_live_hourly_rate("azure_synapse", region="eastus") == 0.163

    def test_facade_never_raises_even_when_cache_explodes(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the outer safety net around unexpected internal errors."""

        def boom() -> PricingCache:
            raise OSError("disk on fire")

        monkeypatch.setattr(live_pricing, "PricingCache", boom)

        assert get_live_hourly_rate("azure_synapse", region="eastus") is None

    def test_facade_handles_non_string_platform(self, cache_path: Path, live_enabled: None) -> None:
        """Test that even a bogus platform argument cannot raise."""
        assert get_live_hourly_rate(None, region="eastus") is None  # type: ignore[arg-type]

    def test_end_to_end_azure_with_mock_transport(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the full facade -> httpx path with a MockTransport."""
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"Items": [_azure_item(0.163)]}),
        )
        monkeypatch.setattr(
            live_pricing,
            "AzureRetailPricesClient",
            lambda: AzureRetailPricesClient(transport=transport),
        )

        assert get_live_hourly_rate("azure_synapse", region="westeurope") == 0.163

    def test_missing_boto3_returns_none(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a missing boto3 results in None (and a negative entry)."""
        monkeypatch.setitem(sys.modules, "boto3", None)

        assert get_live_hourly_rate("aws_glue", region="us-east-1") is None
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached["aws_glue|us-east-1|"]["rate"] is None


class TestGetLiveHourlyRateGCP:
    """Test cases for the gcp_dataproc path of the live pricing facade."""

    @pytest.fixture
    def gcp_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provide a GCP API key for the duration of a test."""
        monkeypatch.setenv(GCP_API_KEY_ENV_VAR, "test-key")

    def _patch_gcp(self, monkeypatch: pytest.MonkeyPatch, result: Any) -> MagicMock:
        """Replace the GCP client class with a mock returning ``result``."""
        instance = MagicMock()
        if isinstance(result, BaseException):
            instance.get_n2_machine_rate.side_effect = result
        else:
            instance.get_n2_machine_rate.return_value = result
        monkeypatch.setattr(live_pricing, "GCPCloudBillingClient", MagicMock(return_value=instance))
        return instance

    def test_disabled_returns_none_even_with_key(
        self,
        cache_path: Path,
        gcp_key: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that the default-off path stays inert despite a key being set."""
        monkeypatch.delenv("SPARK_OPTIMA_LIVE_PRICING", raising=False)
        fetch = MagicMock()
        monkeypatch.setattr(live_pricing, "_fetch_rate", fetch)

        rate = get_live_hourly_rate(
            "gcp_dataproc",
            region="us-central1",
            instance_type="n2-standard-8",
            vcpus=8,
            memory_gb=32,
        )

        assert rate is None
        fetch.assert_not_called()
        assert not cache_path.exists()

    def test_no_api_key_returns_none_without_fetch_or_cache(
        self,
        cache_path: Path,
        live_enabled: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the env gate: no SPARK_OPTIMA_GCP_API_KEY means no lookup at all."""
        monkeypatch.delenv(GCP_API_KEY_ENV_VAR, raising=False)
        fetch = MagicMock()
        monkeypatch.setattr(live_pricing, "_fetch_rate", fetch)

        rate = get_live_hourly_rate(
            "gcp_dataproc",
            region="us-central1",
            instance_type="n2-standard-8",
            vcpus=8,
            memory_gb=32,
        )

        assert rate is None
        fetch.assert_not_called()
        assert not cache_path.exists()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"vcpus": 8, "memory_gb": 32},  # Missing instance_type
            {"instance_type": "n2-standard-8", "memory_gb": 32},  # Missing vcpus
            {"instance_type": "n2-standard-8", "vcpus": 8},  # Missing memory_gb
            {"instance_type": "n2-standard-8", "vcpus": 0, "memory_gb": 32},  # Non-positive vcpus
            {"instance_type": "n2-standard-8", "vcpus": 8, "memory_gb": -1},  # Non-positive memory
        ],
    )
    def test_missing_machine_specs_return_none(
        self,
        cache_path: Path,
        live_enabled: None,
        gcp_key: None,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict[str, Any],
    ) -> None:
        """Test that incomplete machine specs skip the lookup entirely."""
        fetch = MagicMock()
        monkeypatch.setattr(live_pricing, "_fetch_rate", fetch)

        assert get_live_hourly_rate("gcp_dataproc", region="us-central1", **kwargs) is None
        fetch.assert_not_called()

    def test_dispatch_and_cache_key_by_machine_type(
        self,
        cache_path: Path,
        live_enabled: None,
        gcp_key: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the happy path: client dispatch, rate, and cache key shape."""
        instance = self._patch_gcp(monkeypatch, 0.388472)

        rate = get_live_hourly_rate(
            "gcp_dataproc",
            region="us-central1",
            instance_type="n2-standard-8",
            vcpus=8,
            memory_gb=32,
        )

        assert rate == pytest.approx(0.388472)
        instance.get_n2_machine_rate.assert_called_once_with("us-central1", vcpus=8.0, memory_gb=32.0)
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached["gcp_dataproc|us-central1|n2-standard-8"]["rate"] == pytest.approx(0.388472)

    def test_second_call_served_from_cache(
        self,
        cache_path: Path,
        live_enabled: None,
        gcp_key: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a fresh cache entry avoids a second fetch."""
        instance = self._patch_gcp(monkeypatch, 0.388472)

        first = get_live_hourly_rate(
            "gcp_dataproc",
            region="us-central1",
            instance_type="n2-standard-8",
            vcpus=8,
            memory_gb=32,
        )
        second = get_live_hourly_rate(
            "gcp_dataproc",
            region="us-central1",
            instance_type="n2-standard-8",
            vcpus=8,
            memory_gb=32,
        )

        assert first == second == pytest.approx(0.388472)
        assert instance.get_n2_machine_rate.call_count == 1

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectTimeout("timed out"),
            httpx.HTTPStatusError("403", request=MagicMock(), response=MagicMock()),
            LookupError("no N2 SKUs"),
            ValueError("bad json"),
            RuntimeError("API key required"),
        ],
    )
    def test_any_fetch_failure_returns_none(
        self,
        cache_path: Path,
        live_enabled: None,
        gcp_key: None,
        monkeypatch: pytest.MonkeyPatch,
        error: BaseException,
    ) -> None:
        """Test that every fetch failure mode yields None, never an exception."""
        self._patch_gcp(monkeypatch, error)

        rate = get_live_hourly_rate(
            "gcp_dataproc",
            region="us-central1",
            instance_type="n2-standard-8",
            vcpus=8,
            memory_gb=32,
        )

        assert rate is None

    def test_failure_is_negative_cached(
        self,
        cache_path: Path,
        live_enabled: None,
        gcp_key: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a failed GCP lookup is cached and not retried immediately."""
        instance = self._patch_gcp(monkeypatch, httpx.ConnectTimeout("timed out"))
        kwargs: dict[str, Any] = {
            "region": "us-central1",
            "instance_type": "n2-standard-8",
            "vcpus": 8,
            "memory_gb": 32,
        }

        assert get_live_hourly_rate("gcp_dataproc", **kwargs) is None
        assert get_live_hourly_rate("gcp_dataproc", **kwargs) is None

        assert instance.get_n2_machine_rate.call_count == 1  # Second call hit the negative entry
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached["gcp_dataproc|us-central1|n2-standard-8"]["rate"] is None

    def test_deadline_exceeded_returns_none_and_is_negative_cached(
        self,
        cache_path: Path,
        live_enabled: None,
        gcp_key: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that an over-deadline lookup yields None plus a negative cache entry.

        Uses a monkeypatched clock (no real sleeping): the very first page
        consumes the whole lookup budget, so the client gives up and the
        facade records the failure like any other.
        """
        requests: list[httpx.Request] = []
        clock = {"now": 0.0}

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            clock["now"] += GCPCloudBillingClient.LOOKUP_DEADLINE_SECONDS  # Each page blows the budget
            return httpx.Response(200, json={"skus": [_gcp_sku("Unrelated SKU")], "nextPageToken": "more"})

        monkeypatch.setattr(live_pricing, "time", SimpleNamespace(monotonic=lambda: clock["now"], time=time.time))
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            live_pricing,
            "GCPCloudBillingClient",
            lambda: GCPCloudBillingClient(api_key="test-key", transport=transport),
        )

        rate = get_live_hourly_rate(
            "gcp_dataproc",
            region="us-central1",
            instance_type="n2-standard-8",
            vcpus=8,
            memory_gb=32,
        )

        assert rate is None
        assert len(requests) == 1  # Gave up right after the first slow page
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached["gcp_dataproc|us-central1|n2-standard-8"]["rate"] is None

    def test_end_to_end_gcp_with_mock_transport(
        self,
        cache_path: Path,
        live_enabled: None,
        gcp_key: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the full facade -> httpx path with a MockTransport."""
        transport = _gcp_transport_for_pages([[_gcp_core_sku(), _gcp_ram_sku()]])
        monkeypatch.setattr(
            live_pricing,
            "GCPCloudBillingClient",
            lambda: GCPCloudBillingClient(api_key="test-key", transport=transport),
        )

        rate = get_live_hourly_rate(
            "gcp_dataproc",
            region="us-central1",
            instance_type="n2-standard-8",
            vcpus=8,
            memory_gb=32,
        )

        assert rate == pytest.approx(8 * 0.031611 + 32 * 0.004237)
