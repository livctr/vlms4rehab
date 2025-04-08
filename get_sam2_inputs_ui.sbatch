#!/bin/bash
#SBATCH --partition gpu4_dev
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 16G
#SBATCH --gres gpu:1
#SBATCH --time 0-04:00:00

# Thank you, Yale Center for Research Computing, for this script
# modified for use on BigPurple in NYU Langone Medical Center by Paul Glick

# get tunneling info
XDG_RUNTIME_DIR=""
export FLASK_PORT=9172
export FRONTEND_PORT=9173
export NODE=$(hostname -s)
export USER=$(whoami)


case "$SLURM_SUBMIT_HOST" in
bigpurple-ln1)
	LOGIN_NODE=bigpurple1.nyumc.org
	;;
bigpurple-ln2)
	LOGIN_NODE=bigpurple2.nyumc.org
	;;
bigpurple-ln3)
	LOGIN_NODE=bigpurple3.nyumc.org
	;;
bigpurple-ln4)
	LOGIN_NODE=bigpurple4.nyumc.org
	;;
esac
# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${FRONTEND_PORT}:${NODE}:${FRONTEND_PORT} ${USER}@${LOGIN_NODE} 

For more info and how to connect from windows, 
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Remote server: ${NODE}
Remote frontend port: ${FRONTEND_PORT}
SSH server: ${LOGIN_NODE}
SSH login: ${USER}
SSH port: 22
Use a Browser on your local machine to go to:
http://localhost:${FRONTEND_PORT}
"

# load modules or conda environments here

source ~/.bashrc
conda init
cd /gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/victor/cvfm4rehab
conda activate sam2  # segmentation fault in cvfm4rehab? lazy to redo it

# Start the backend in the background and store its PID
echo "Starting backend on ${NODE}:${FLASK_PORT}"
python -m data.pipeline.backend.server.app --port ${FLASK_PORT} --ip ${NODE} &
BACKEND_PID=$!

# Wait until the backend port is open (using netcat)
while ! nc -z ${NODE} ${FLASK_PORT}; do
    echo "Waiting for backend to start..."
    sleep 1
done

# Now start the frontend
echo "Starting frontend on ${NODE}:${FRONTEND_PORT}"
cd data/pipeline/frontend
HOST=${NODE} PORT=${FRONTEND_PORT} npm start &
FRONTEND_PID=$!
cd ../../..

echo "Set up complete. Connect!"

# Wait for both processes to finish
wait $BACKEND_PID $FRONTEND_PID
