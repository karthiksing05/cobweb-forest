#!/bin/bash
# Usage: SNAP=$(scripts/snapshot.sh <name>)  -- freeze package + experiments for a sweep
set -e
d=runs/$1-$(date +%m%d-%H%M%S)
mkdir -p $d
cp -r cobweb_forest experiments $d/
rm -rf $d/cobweb_forest/__pycache__ $d/experiments/__pycache__
echo $PWD/$d
