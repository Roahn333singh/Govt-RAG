import pytest


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_not_public(client, path):
    assert client.get(path).status_code == 404
