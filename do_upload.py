import os
import sys
import subprocess

def main():
    pypi_password = os.environ.get("PYPI_PASSWORD", "")
    if not pypi_password:
        print("Error: PYPI_PASSWORD is not set!")
        sys.exit(1)

    print("Uploading version 0.9.5 to PyPI...")
    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = pypi_password

    # Run twine upload on the 0.9.5 wheels/tarballs
    # Need to match specifically the newly built dists, or just upload dist/*
    # Note: If dist/ contains old versions like 0.9.4, twine might skip them if they exist or fail.
    # To be safe, we'll upload specifically sagan_trade-0.9.5*
    import glob
    files = glob.glob("dist/sagan_trade-0.9.5*")
    
    if not files:
        print("No files found for 0.9.5 in dist/!")
        sys.exit(1)
        
    cmd = [sys.executable, "-m", "twine", "upload", "--skip-existing"] + files
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("Error uploading:")
        print(res.stderr)
        sys.exit(1)
    print(f"Successfully published sagan-trade 0.9.5 to PyPI!")

if __name__ == '__main__':
    main()
