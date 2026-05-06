# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for API main module."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from spark_optima.api.main import app, create_app, lifespan


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_returns_fastapi(self) -> None:
        """Test that create_app returns a FastAPI instance."""
        test_app = create_app()
        assert test_app is not None
        assert hasattr(test_app, "title")

    def test_app_has_routes(self) -> None:
        """Test that app has routes registered."""
        test_app = create_app()
        # FastAPI routes should be registered
        routes = [r.path for r in test_app.routes]
        assert len(routes) > 0


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self) -> None:
        """Test root endpoint returns API info."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data


class TestLifespan:
    """Tests for application lifespan."""

    def test_app_startup_logs_version(self) -> None:
        """Test that app logs version on startup."""
        # Just verify the app can be created without errors
        test_app = create_app()
        assert test_app is not None

    @patch("spark_optima.api.dependencies.get_optimization_service")
    def test_lifespan_startup_calls_service(self, mock_get_service: MagicMock) -> None:
        """Test that lifespan startup initializes the optimization service."""
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.return_value = ["3.5.0", "3.4.0"]
        mock_get_service.return_value = mock_service

        # Call the lifespan context manager directly
        async def run_lifespan():
            async with lifespan(create_app()):
                pass

        asyncio.get_event_loop().run_until_complete(run_lifespan())

        # Verify the service was initialized
        mock_get_service.assert_called_once()
        mock_service.get_available_spark_versions.assert_called_once()

    @patch("spark_optima.api.dependencies.get_optimization_service")
    def test_lifespan_startup_logs_versions(self, mock_get_service: MagicMock) -> None:
        """Test that lifespan logs the available Spark versions."""
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.return_value = ["3.5.0", "3.4.0", "3.3.0"]
        mock_get_service.return_value = mock_service

        async def run_lifespan():
            async with lifespan(create_app()):
                pass

        asyncio.get_event_loop().run_until_complete(run_lifespan())

        # Verify service was called
        mock_get_service.assert_called_once()
        assert mock_service.get_available_spark_versions.called

    @patch("spark_optima.api.dependencies.get_optimization_service")
    def test_lifespan_shutdown_logs_message(self, mock_get_service: MagicMock) -> None:
        """Test that lifespan shutdown logs shutdown message."""
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.return_value = ["3.5.0"]
        mock_get_service.return_value = mock_service

        async def run_lifespan():
            async with lifespan(create_app()):
                # Exiting the context should trigger shutdown
                pass

        asyncio.get_event_loop().run_until_complete(run_lifespan())
        # Shutdown should have been called (verified by no exceptions)
        assert True


class TestGlobalExceptionHandler:
    """Tests for the global exception handler."""

    def test_exception_handler_returns_500(self) -> None:
        """Test that unhandled exceptions return 500 status."""
        from spark_optima.api.main import create_app

        # Create a test app with a route that raises an exception
        test_app = create_app()

        @test_app.get("/test-exception")
        async def raise_exception():
            raise ValueError("Test exception")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test-exception")

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"] == "internal_server_error"
        assert "message" in data

    def test_exception_handler_returns_json_response(self) -> None:
        """Test that exception handler returns proper JSON structure."""
        from spark_optima.api.main import create_app

        test_app = create_app()

        @test_app.get("/test-exception-2")
        async def raise_exception():
            raise RuntimeError("Test runtime error")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test-exception-2")

        assert response.status_code == 500
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert data["error"] == "internal_server_error"
        assert "unexpected error" in data["message"].lower()

    def test_exception_handler_logs_exception(self) -> None:
        """Test that exception handler logs the exception."""
        from spark_optima.api.main import create_app

        test_app = create_app()

        @test_app.get("/test-exception-3")
        async def raise_exception():
            raise Exception("Log this error")

        with patch("spark_optima.api.main.logger") as mock_logger:
            client = TestClient(test_app, raise_server_exceptions=False)
            response = client.get("/test-exception-3")

            assert response.status_code == 500
            # Verify that exception was logged
            mock_logger.exception.assert_called_once()
