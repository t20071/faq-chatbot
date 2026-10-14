import faqbot.model_files as mf


def fake_download(calls):
    def urlretrieve(url, path):
        calls.append(url)
        path.write_text("x")

    return urlretrieve


def test_downloads_missing_files_pinned_to_revision(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mf.urllib.request, "urlretrieve", fake_download(calls))

    model_dir = mf.ensure_model_files(tmp_path / "model")

    assert sorted(p.name for p in model_dir.iterdir()) == sorted(mf.FILES)
    assert len(calls) == 2
    assert all(mf.EMBEDDING_MODEL_REVISION in url for url in calls)


def test_existing_files_are_not_downloaded_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mf.urllib.request, "urlretrieve", fake_download(calls))
    for name in mf.FILES:
        (tmp_path / name).write_text("x")

    assert mf.ensure_model_files(tmp_path) == tmp_path
    assert calls == []


def test_read_only_dir_falls_back_to_temp(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mf.urllib.request, "urlretrieve", fake_download(calls))
    monkeypatch.setattr(mf, "_writable", lambda d: False)
    monkeypatch.setattr(mf.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    model_dir = mf.ensure_model_files(tmp_path / "read-only" / "all-MiniLM-L6-v2")

    assert model_dir == tmp_path / "tmp" / "models" / "all-MiniLM-L6-v2"
    assert len(calls) == 2
