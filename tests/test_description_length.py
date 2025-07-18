import pathlib

def test_description_length_constant():
    path = pathlib.Path('analysis/data_preparation/01_import_filter_orgs.py')
    content = path.read_text()
    assert "str.len() >= 150" in content
