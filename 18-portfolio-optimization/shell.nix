{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
in
pkgs.mkShell {
  buildInputs = [
    python
    python.pkgs.pip
    python.pkgs.setuptools
    pkgs.zlib
    pkgs.glib
  ];

  # Assure que les libs natives comme libz.so.1 sont visibles
  LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [
    zlib
    stdenv.cc.cc
    glib
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi

    # Activation venv
    if [ -f ".venv/bin/activate" ]; then
      source .venv/bin/activate
    fi

    # Hack pour que venv trouve les libs partagées de Nix
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH


    # Install deps si pas déjà fait
    if [ ! -f ".venv/.installed" ]; then
      pip install -r requirements.txt && touch .venv/.installed
    fi
  '';
}
