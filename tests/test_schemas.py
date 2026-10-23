import pytest

from greenshaping.data.schemas import DOWN, UP, Message, Session, SessionManifest, size_bucket, size_bucket_clipped


@pytest.mark.parametrize("rows,clipped", [(999, True), (1000, False), (999_999, False),
                                         (1_000_000, True), (9_999_999, True), (10_000_000, True)])
def test_clipping_matches_manuscript_half_open_range(rows, clipped):
    assert size_bucket_clipped(rows) is clipped


def test_size_bucket_edges():
    assert size_bucket(1000) == 0 and size_bucket(9999) == 0
    assert size_bucket(10_000) == 1 and size_bucket(99_999) == 1
    assert size_bucket(100_000) == 2 and size_bucket(1_000_000) == 2
    assert size_bucket(0) == 0          # called site with no matching rows: clipped
    with pytest.raises(ValueError):
        size_bucket(-1)


def test_message_validation():
    with pytest.raises(ValueError):
        Message(0.0, 0, 7, 10)
    with pytest.raises(ValueError):
        Message(0.0, 0, UP, -1)


def test_session_label_validation():
    with pytest.raises(ValueError):
        Session(0, 0, 2, 0, [1, 0], [0, 0])          # non-participant must have bucket -1
    with pytest.raises(ValueError):
        Session(0, 0, 2, 9, [1, 0], [0, -1])         # bad query family
    with pytest.raises(ValueError):
        Session(0, 0, 1, 0, [1], [0], [Message(0.0, 3, UP, 1)])   # link out of range


def test_session_roundtrip(session_factory):
    s = session_factory([500, 0], [900, 0])
    s2 = Session.from_dict(s.to_dict())
    assert s2.messages == s.messages and s2.participation == s.participation


def test_manifest_requires_known_family():
    d = {"session_id": 1, "question_id": 2, "query_family": "nope", "start_time": 0.0,
         "sites": [{"ip": "10.0.0.1", "port": 1, "participates": 0}]}
    with pytest.raises(ValueError):
        SessionManifest.from_dict(d)
    d["query_family"] = "aggregate"
    m = SessionManifest.from_dict(d)
    assert m.sites[0].bucket() == -1
    assert DOWN == 1
