import subprocess
import sys
import os

def run(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}\n{result.stderr}")
        sys.exit(result.returncode)
    return result.stdout.strip()

def main():
    # Ensure we are in the repo root (assume this script is in ./scripts)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)
    # Initialize git repo if needed
    if not os.path.isdir('.git'):
        run('git init')
        run('git add .')
        run('git commit -m "Initial commit for sagan-trade release"')
    # Set remote if not set
    remotes = run('git remote').splitlines()
    pat = os.environ.get('GITHUB_PAT')
    if 'origin' in remotes:
        run('git remote remove origin')
    run(f'git remote add origin https://That-Tech-Geek:{pat}@github.com/That-Tech-Geek/sagan-trade.git')
    # Tag version
    version = "0.3.0"
    tag_name = f"v{version}"
    # Delete existing tag locally/remotely if present
    existing_tags = run('git tag').splitlines()
    if tag_name in existing_tags:
        run(f'git tag -d {tag_name}')
        run(f'git push origin :refs/tags/{tag_name}', cwd=repo_root)  # delete remote tag
    run(f'git tag {tag_name}')
    # Push everything
    run('git push -u origin main --tags')

if __name__ == "__main__":
    main()
