import pytest

from rag_production.config import IngestAuthError, authorize_ingest, load_settings


def test_ingest_fails_closed_when_key_unset():
    with pytest.raises(IngestAuthError) as e:
        authorize_ingest(load_settings({}), "anything")
    assert e.value.status == 503


@pytest.mark.parametrize("provided", [None, "", "wrong"])
def test_ingest_rejects_missing_or_wrong_key(provided):
    with pytest.raises(IngestAuthError) as e:
        authorize_ingest(load_settings({"API_KEY": "s3cret-test"}), provided)
    assert e.value.status == 401


def test_ingest_accepts_matching_key():
    authorize_ingest(load_settings({"API_KEY": "s3cret-test"}), "s3cret-test")


def test_settings_repr_does_not_leak_api_key():
    s = load_settings({"API_KEY": "s3cret-test", "DATABASE_URL": "postgresql://u:pw123@h/db"})
    assert "s3cret-test" not in repr(s)
    assert "pw123" not in repr(s)
