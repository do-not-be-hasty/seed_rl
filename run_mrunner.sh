#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./scripts/run_mrunner [project_tags]"
    exit 1
fi

source ../../venvs/py36_seedrl/bin/activate

export NEPTUNE_PROJECT_NAME="do-not-be-hasty/tmp"

ssh-add
./basic_setup.sh plgmizaw sim2real2

cd ..

if [ ! -z "$1" ]; then
        export PROJECT_TAG="$1"
fi

echo "Run experiments"
set -o xtrace
mrunner --config /tmp/mrunner_config.yaml --context entropy_cpu run seed_rl/run_conf.py
set +o xtrace
