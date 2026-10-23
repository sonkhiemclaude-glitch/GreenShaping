from greenshaping.data.synthetic import SyntheticConfig, generate_sessions


def test_generator_deterministic_and_valid():
    a = generate_sessions(SyntheticConfig(n_sessions=30, n_sites=5, n_questions=12, seed=3))
    b = generate_sessions(SyntheticConfig(n_sessions=30, n_sites=5, n_questions=12, seed=3))
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    for s in a:
        s.validate()
        assert sum(s.participation) >= 1
        assert s.meta["data_kind"] == "synthetic"
        called = {m.site for m in s.messages}
        assert called == {i for i, p in enumerate(s.participation) if p}


def test_different_seed_differs():
    a = generate_sessions(SyntheticConfig(n_sessions=10, n_sites=3, n_questions=8, seed=1))
    b = generate_sessions(SyntheticConfig(n_sessions=10, n_sites=3, n_questions=8, seed=2))
    assert [s.to_dict() for s in a] != [s.to_dict() for s in b]
