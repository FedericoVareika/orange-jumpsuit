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
    xorg.libX11
    xorg.libXi
    xorg.libXrender
    xorg.libICE
    xorg.libSM
    xorg.libxcb
    xorg.libXext
    
    # NEW: Libraries to fix the Font/Qt error
    fontconfig
    freetype
    libxkbcommon  # Fixes potential keyboard crash in Qt windows
    
    # Tcl/Tk for Matplotlib backend
    tcl
    tk

    pkgs.openblas
  ];
in
pkgs.mkShell {
  name = "python-data-science";
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
    
    # 2. Point LD_LIBRARY_PATH to all the libs defined above
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath extraLibs}:$LD_LIBRARY_PATH
    
    # 3. NEW: Tell Qt/OpenCV where to find system fonts
    export FONTCONFIG_FILE=/etc/fonts/fonts.conf
    
    echo "==================================================="
    echo "  Data Science Environment Ready"
    echo "  - Jupyter: run 'jupyter lab'"
    echo "  - VS Code: run 'code .'"
    echo "==================================================="
  '';
}
