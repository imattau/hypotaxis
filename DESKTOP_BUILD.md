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

The Linux workflow publishes `hypotaxis-linux-packages` containing an AppImage,
Debian package, and RPM package, plus a separate `hypotaxis-flatpak` artifact.
Version tags also
publish APT and RPM repository indexes to `imattau/hypotaxis-distribution` via
GitHub Pages. Configure a repository secret named `DISTRIBUTION_REPO_TOKEN`
with write access to that repository before using tagged releases.
Tagged publishing also requires `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`, and
`GPG_KEY_ID`; the workflow validates all four secrets before downloading or
indexing packages.

The Flatpak job depends only on the Linux package job's executable input, not on
the complete cross-platform matrix. This keeps Flatpak available when a
Windows or macOS runner is delayed.

Pushes to `master`, feature branches, tags beginning with `v`, and pull
requests trigger the workflow. The generated artifact is a platform-native
PyInstaller application; it should be tested on the target operating system
before release. Tagged GitHub releases also include a `SHA256SUMS` manifest
covering every downloaded build and package artifact.

The workflow cancels superseded runs for the same ref, preventing older queued
platform jobs from accumulating behind newer commits.

Windows users also need a supported Microsoft WebView2 Runtime installed on
the target machine. Linux users need the Qt or GTK native libraries required by
the selected pywebview2 backend.

## Linux package repository

The public package repository is published at
`https://imattau.github.io/hypotaxis-distribution/` after a version tag is
built. Debian/Ubuntu users can add:

```bash
curl -fsSL https://imattau.github.io/hypotaxis-distribution/deb/hypotaxis-archive-keyring.asc \
  | sudo tee /usr/share/keyrings/hypotaxis-archive-keyring.asc >/dev/null
echo "deb [signed-by=/usr/share/keyrings/hypotaxis-archive-keyring.asc] https://imattau.github.io/hypotaxis-distribution/deb stable main" \
  | sudo tee /etc/apt/sources.list.d/hypotaxis.list >/dev/null
sudo apt update && sudo apt install hypotaxis
```

RPM-based users can create a repo file pointing at:

```text
https://imattau.github.io/hypotaxis-distribution/rpm/
```

The APT `Release` file and RPM repository metadata are signed with the
Hypotaxis release key. The public key is published alongside the Debian
repository.
