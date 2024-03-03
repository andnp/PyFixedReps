#!/bin/bash
set -e

git config credential.helper "store --file=.git/credentials"
echo "https://${GH_TOKEN}:@github.com" > .git/credentials

git config user.email "andnpatterson@gmail.com"
git config user.name "github-action"

git fetch --all --tags

git checkout -f main

# bump the version
cz bump --no-verify --yes --check-consistency

pip install uv
uv pip compile --extra=dev pyproject.toml -o requirements.txt
git add requirements.txt
git commit -m "ci: update requirements" || echo "No changes to commit"
