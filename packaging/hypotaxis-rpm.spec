Name:           hypotaxis
Version:        0.0.0
Release:        1
Summary:        Local manga production studio
License:        Proprietary
BuildArch:      x86_64

%description
Hypotaxis is a local-first manga production studio.

%install
install -D -m 0755 %{_sourcedir}/hypotaxis %{buildroot}/opt/hypotaxis/hypotaxis
install -D -m 0644 %{_sourcedir}/hypotaxis.desktop %{buildroot}/usr/share/applications/hypotaxis.desktop
install -D -m 0644 %{_sourcedir}/hypotaxis.svg %{buildroot}/usr/share/icons/hicolor/scalable/apps/hypotaxis.svg

%files
/opt/hypotaxis/hypotaxis
/usr/share/applications/hypotaxis.desktop
/usr/share/icons/hicolor/scalable/apps/hypotaxis.svg
