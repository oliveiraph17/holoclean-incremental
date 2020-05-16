CLEANCMD="rm -f /tmp/checkpoint* && rm -f /tmp/*.ujson"
PYCMD="python incremental_script.py"
HOSPITAL_SHUFFLED="hospital_shuffled _tid_ 10 100 0.99 10000 0.6 0.8 dk"
FOOD5K_SHUFFLED="food5k_shuffled _tid_ 50 100 0.6 500 0.2 0.3 dk"
NYPD6="nypd6 _ 324 100 0.9 60 0.05 0.3 dk"
SOCCER="soccer _ 2000 100 0.9 40 0.05 0.3 dk"

DATASET=$FOOD5K_SHUFFLED
APPROACH="B"

echo -n "source ../set_env.sh && "

LOGFILE=$(echo "$APPROACH $DATASET" | sed s/' '/'-'/g)
echo -n "$CLEANCMD && $PYCMD $APPROACH $DATASET > $LOGFILE.log 2> $LOGFILE.err && "
    
echo "echo Finished!"

