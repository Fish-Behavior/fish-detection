from backend.shared.scripts.changed_tests import select


def test_select_maps_changed_files_to_units():
    assert select(["backend/dcs/src/dcs/gold.py"]) == ["dcs"]
    assert select(["backend/prepds/docs/progress.md", "README.md", "frontend/src/a.ts"]) == []
    assert select(["backend/prepds/tests/x.py", "backend/dcs/config/a.yaml"]) == ["dcs", "prepds"]
    assert select(["pyproject.toml"]) == ["dcs", "prepds", "shared"]
    assert select(["backend/shared/tests/fixtures/a.png"]) == ["dcs", "prepds", "shared"]
