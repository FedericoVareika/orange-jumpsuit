{ pkgs ? import <nixpkgs> {} }:

let
  # 1. Define libraries required by OpenCV/Matplotlib pip wheels
  extraLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
    libGL
    libGLU
    
    # X11 / GUI Libraries
    libX11
    libXi
    libXrender
    libICE
    libSM
    libxcb
    libXext
    
    # NEW: Libraries to fix the Font/Qt error
    fontconfig
    freetype
    libxkbcommon  # Fixes potential keyboard crash in Qt windows

    tcl
    tk
  ];
in
pkgs.mkShell {
  name = "python-build-jumpsuit";
  venvDir = "./.venv";

  buildInputs = [
    # Python with Tkinter enabled
    (pkgs.python3.withPackages (ps: [ ps.tkinter ]))
    pkgs.python3Packages.venvShellHook
  ] ++ extraLibs;

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH

    pip install -U pip setuptools wheel
    
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi
  '';

  postShellHook = ''
    unset SOURCE_DATE_EPOCH

    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath extraLibs}:$LD_LIBRARY_PATH
    
    # 3. NEW: Tell Qt/OpenCV where to find system fonts
    export FONTCONFIG_FILE=/etc/fonts/fonts.conf
  '';
}
