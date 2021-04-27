#!/bin/bash
mkdir -p ./build/gcd
mkdir ./local_server_work
GCD_DIR=`dirname "$0"`

# Start a local sc-server in a background task.
sc-server -nfs_mount ./local_server_work -cluster local &
SERVER_PID=$!

# Run a remote sc job targeting localhost as the 'remote'.
sc $GCD_DIR/gcd.v \
  -design gcd \
  -target freepdk45 \
  -asic_diesize "0 0 100.13 100.8" \
  -asic_coresize "10.07 11.2 90.25 91" \
  -constraint $GCD_DIR/constraint.sdc \
  -remote localhost

# Kill the temporary local sc-server process.
kill $SERVER_PID

# Get the job hash directory name; we know that it will be a hex string.
JOB_HASH=`find . -maxdepth 1 -regex "\.\/[0-9a-f]*" | sed 's/\.\///g'`
if [ -f "./$JOB_HASH/gcd/job1/export/outputs/gcd.gds" ]; then
  echo "Success!"
  exit 0
fi
echo "Fail :("
exit 1
