{
    description = "Orange Jumpsuit development";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    };

    outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        # The packages you need available in the shell
        packages = with pkgs; [
          gcc
          openblas
        ];

        # Optional: Print a message when entering the shell
        shellHook = ''
          echo "Orange Jumpsuit Dev Environment Loaded!"
        '';
      };
    };
}

