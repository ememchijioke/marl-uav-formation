#!/bin/bash

echo "========================================"
echo "PROJECT INFORMATION"
echo "========================================"

echo -e "\nCurrent Directory:"
pwd

echo -e "\nGit Branch:"
git branch --show-current

echo -e "\nGit Status:"
git status --short

echo -e "\nGit Remote:"
git remote -v

echo -e "\nLast 5 Commits:"
git log --oneline -5

echo -e "\n========================================"
echo "PROJECT STRUCTURE"
echo "========================================"

if command -v tree >/dev/null 2>&1; then
    tree -L 4 -I "__pycache__|.git|venv|.venv|build|dist|*.egg-info|wandb|videos|results|checkpoints"
else
    find . -maxdepth 4 \
      -not -path "*/.git/*" \
      -not -path "*/__pycache__/*" \
      -not -path "*/venv/*" \
      -not -path "*/.venv/*" \
      | sort
fi
