# Desktop builds

Hypotaxis can run as a native desktop window while keeping the existing FastAPI
studio server as its local UI backend.

## Local development

```bash
python -m pip install -r requirements-desktop.txt
python run_desktop.py
```

`--port 0` (the default) chooses a free loopback port. The launcher shuts down
the embedded Uvicorn server when the native window closes.

## GitHub Actions builds

The `Build desktop applications` workflow builds independently on Ubuntu,
Windows, and macOS using PyInstaller. Each run publishes these artifacts:

- `hypotaxis-linux`
- `hypotaxis-windows`
- `hypotaxis-macos`

The Linux job additionally publishes `hypotaxis-linux-packages` containing a
Flatpak bundle, AppImage, Debian package, and RPM package. Version tags also
publish APT and RPM repository indexes to `imattau/hypotaxis-distribution` via
GitHub Pages. Configure a repository secret named `DISTRIBUTION_REPO_TOKEN`
with write access to that repository before using tagged releases.

Pushes to `master`, feature branches, tags beginning with `v`, and pull
requests trigger the workflow. The generated artifact is a platform-native
PyInstaller application; it should be tested on the target operating system
before release.

Windows users also need a supported Microsoft WebView2 Runtime installed on
the target machine. Linux users need the Qt or GTK native libraries required by
the selected pywebview2 backend.

## Linux package repository

The public package repository is published at
`https://imattau.github.io/hypotaxis-distribution/` after a version tag is
built. Debian/Ubuntu users can add:

```text
deb [trusted=yes] https://imattau.github.io/hypotaxis-distribution/deb stable main
```

RPM-based users can create a repo file pointing at:

```text
https://imattau.github.io/hypotaxis-distribution/rpm/
```
