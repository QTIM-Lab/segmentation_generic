#!/bin/sh

if [ -n "$1" ]; then
  echo "linting \"$1\""
fi

echo "running flake8"
if [ -n "$1" ]; then
  flake8 "$1"
else
  flake8 src
fi

echo "running mypy"
if [ -n "$1" ]; then
  mypy "$1"
else
  mypy src
fi

exit 0