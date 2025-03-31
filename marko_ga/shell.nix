{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.python312Packages.setuptools
    pkgs.stdenv.cc.cc.lib
  ];
  LD_LIBRARY_PATH="${pkgs.libGL}/lib/:${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.glib.out}/lib/";

  shellHook = ''
    if [ ! -d ".venv" ]; then
      python3 -m venv .venv
      . .venv/bin/activate
    fi
    # Activate virtual environment if it exists
    if [ -f ".venv/bin/activate" ]; then
      source .venv/bin/activate
    fi
    pip install -r requirements.txt
  '';
}