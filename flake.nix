{
  description = "Orange Jumpsuit development";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      
      pkgsCrossWindows = import nixpkgs {
        inherit system;
        crossSystem = {
          config = "x86_64-w64-mingw32";
        };
      };
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = with pkgs; [ gcc openblas ];
          shellHook = ''echo "Orange Jumpsuit (Linux) Dev Environment Loaded!"'';
        };

        windows = pkgsCrossWindows.mkShell {
          nativeBuildInputs = [
            pkgsCrossWindows.stdenv.cc
          ];

          buildInputs = [ 
            pkgsCrossWindows.windows.pthreads
            pkgsCrossWindows.openblas 
          ];

          shellHook = ''
            export CC=x86_64-w64-mingw32-gcc
            
            # 1. Find the directory containing libgomp.spec within the GCC store path
            # We look inside the unwrapped CC (.cc) to find the actual library files
            GOMP_PATH=$(find ${pkgsCrossWindows.stdenv.cc.cc} -name "libgomp.spec" -exec dirname {} \; | head -n 1)
            
            if [ -n "$GOMP_PATH" ]; then
              # 2. Use -B to tell GCC to look in that directory for its components.
              # IMPORTANT: The trailing slash is often required by GCC for the -B flag!
              export NIX_CFLAGS_COMPILE="-B$GOMP_PATH/ $NIX_CFLAGS_COMPILE"
              echo "Found libgomp.spec at: $GOMP_PATH"
            else
              echo "Warning: libgomp.spec not found in ${pkgsCrossWindows.stdenv.cc.cc}"
            fi

            echo "Orange Jumpsuit (Windows + OpenMP) Loaded!"
          '';
        };
      };
    };
}

# ... (rest of your flake inputs)

