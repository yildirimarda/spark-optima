# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the opt-in live pricing layer.

This module contains tests for the Azure Retail Prices client, the guarded
boto3 AWS Pricing client, the JSON file cache, and the never-raising
``get_live_hourly_rate`` facade. All HTTP traffic is mocked with
``httpx.MockTransport`` and boto3 is replaced with mocks — no test performs
real network calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import httpx
import pytest

from spark_optima.platforms import live_pricing
from spark_optima.platforms.live_pricing import (
    DEFAULT_TTL_HOURS,
    AWSPricingClient,
    AzureRetailPricesClient,
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

    @pytest.mark.parametrize("platform", ["gcp_dataproc", "databricks", "local", "unknown"])
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
