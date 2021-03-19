#!/bin/bash
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


echo "run begins"
pwd

EXPDIR=`pwd`

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

echo "in DIR $DIR"

ENVIRONMENT=$1
AGENT=$2
NUM_ACTORS=$SLURM_NTASKS
MRUNER_CONFIG=$4
shift 4

echo "args read"

cd ../..
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd $DIR

echo "PYTHONPATH updated"

export PYTHONPATH=$PYTHONPATH:/
# set random seed common across tasks
RANDOM=$SLURM_OB_ID
NONCE=id$RANDOM$RANDOM$RANDOM
#echo "nonce created"

RANDOM_PORT=$((49152 + RANDOM % (65535 - 49152)))
SERVER_ADDRESS_LOCAL="localhost:$RANDOM_PORT"
SERVER_ADDRESS_REMOTE="$SLURM_JOB_NODELIST:$RANDOM_PORT"

echo $SLURM_JOB_NODELIST

ACTOR_BINARY="python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=actor --nonce=${NONCE} --server_address=$SERVER_ADDRESS_LOCAL --mrunner_config=$EXPDIR/$MRUNER_CONFIG";
LEARNER_BINARY="python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=learner --nonce=${NONCE} --server_address=$SERVER_ADDRESS_LOCAL --mrunner_config=$EXPDIR/$MRUNER_CONFIG";

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "-------"
echo $ACTOR_BINARY
echo $LEARNER_BINARY

echo "TASK NO"
echo $SLURM_STEP_ID

echo "SLURM_JOB_NODELIST"
echo $SLURM_JOB_NODELIST

NUM_ACTORS=$((MRUNNER_NTASKS - 1))
echo "NUM_ACTORS"
echo $NUM_ACTORS

if [ 0 -eq $SLURM_STEP_ID ];
then
    echo "Running the learner"
    ${LEARNER_BINARY} --logtostderr --num_envs=${NUM_ACTORS}
else
  TASK_ID=$((SLURM_STEP_ID - 1))
  echo "Running an actor $TASK_ID"
  ${ACTOR_BINARY} --logtostderr --num_envs=${NUM_ACTORS} --task=$TASK_ID
fi

