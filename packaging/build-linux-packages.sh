#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: build-linux-packages.sh VERSION [APPIMAGETOOL]}"
APPIMAGETOOL="${2:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
PACKAGE_DIR="${DIST_DIR}/linux-packages"
rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}"

test -x "${DIST_DIR}/hypotaxis"

# Debian package.
DEB_ROOT="${PACKAGE_DIR}/deb-root"
mkdir -p "${DEB_ROOT}/DEBIAN" "${DEB_ROOT}/opt/hypotaxis" "${DEB_ROOT}/usr/share/applications" "${DEB_ROOT}/usr/share/icons/hicolor/scalable/apps"
install -m 0755 "${DIST_DIR}/hypotaxis" "${DEB_ROOT}/opt/hypotaxis/hypotaxis"
install -m 0644 "${ROOT_DIR}/packaging/hypotaxis.desktop" "${DEB_ROOT}/usr/share/applications/hypotaxis.desktop"
install -m 0644 "${ROOT_DIR}/packaging/hypotaxis.svg" "${DEB_ROOT}/usr/share/icons/hicolor/scalable/apps/hypotaxis.svg"
cat > "${DEB_ROOT}/DEBIAN/control" <<EOF
Package: hypotaxis
Version: ${VERSION}
Section: graphics
Priority: optional
Architecture: amd64
Maintainer: Hypotaxis maintainers
Description: Local manga production studio
 Hypotaxis is a local-first manga production studio.
EOF
dpkg-deb --build "${DEB_ROOT}" "${PACKAGE_DIR}/hypotaxis_${VERSION}_amd64.deb" >/dev/null

# RPM package.
RPM_ROOT="${PACKAGE_DIR}/rpmbuild"
mkdir -p "${RPM_ROOT}/SOURCES" "${RPM_ROOT}/SPECS" "${RPM_ROOT}/BUILD" "${RPM_ROOT}/RPMS" "${RPM_ROOT}/SRPMS"
cp "${DIST_DIR}/hypotaxis" "${RPM_ROOT}/SOURCES/hypotaxis"
cp "${ROOT_DIR}/packaging/hypotaxis.desktop" "${RPM_ROOT}/SOURCES/hypotaxis.desktop"
cp "${ROOT_DIR}/packaging/hypotaxis.svg" "${RPM_ROOT}/SOURCES/hypotaxis.svg"
sed "s/^Version:.*/Version:        ${VERSION}/" "${ROOT_DIR}/packaging/hypotaxis-rpm.spec" > "${RPM_ROOT}/SPECS/hypotaxis.spec"
rpmbuild --define "_topdir ${RPM_ROOT}" -bb "${RPM_ROOT}/SPECS/hypotaxis.spec" >/dev/null
cp "${RPM_ROOT}/RPMS/x86_64/"*.rpm "${PACKAGE_DIR}/"

# AppImage. appimagetool is downloaded by CI and is optional for local deb/rpm builds.
if [[ -n "${APPIMAGETOOL}" ]]; then
    APP_DIR="${PACKAGE_DIR}/Hypotaxis.AppDir"
    mkdir -p "${APP_DIR}/usr/bin" "${APP_DIR}/usr/share/applications" "${APP_DIR}/usr/share/icons/hicolor/scalable/apps"
    install -m 0755 "${DIST_DIR}/hypotaxis" "${APP_DIR}/usr/bin/hypotaxis"
    install -m 0755 "${ROOT_DIR}/packaging/AppRun" "${APP_DIR}/AppRun"
    install -m 0644 "${ROOT_DIR}/packaging/hypotaxis.desktop" "${APP_DIR}/hypotaxis.desktop"
    install -m 0644 "${ROOT_DIR}/packaging/hypotaxis.svg" "${APP_DIR}/hypotaxis.svg"
    install -m 0644 "${ROOT_DIR}/packaging/hypotaxis.desktop" "${APP_DIR}/usr/share/applications/hypotaxis.desktop"
    install -m 0644 "${ROOT_DIR}/packaging/hypotaxis.svg" "${APP_DIR}/usr/share/icons/hicolor/scalable/apps/hypotaxis.svg"
    APPIMAGE_EXTRACT_AND_RUN=1 "${APPIMAGETOOL}" "${APP_DIR}" "${PACKAGE_DIR}/Hypotaxis-${VERSION}-x86_64.AppImage"
fi

find "${PACKAGE_DIR}" -maxdepth 1 -type f -printf '%f\n' | sort
