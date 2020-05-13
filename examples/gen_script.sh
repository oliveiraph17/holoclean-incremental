CLEANCMD="rm -f /tmp/checkpoint* && rm -f /tmp/*.ujson"
PYCMD="python incremental_script.py"
KL_THRESH="0.1 0.05 0.01"
KL_STRATEGY="weighted_kl individual_kl _"
GROUPING_THRESH=0.97
SIM_THRESH=0.005
GROUPING_STRATEGY="_ pair_corr corr_sim"
HOSPITAL_SHUFFLED="C hospital_shuffled _tid_ 10 100 0.99 10000 0.6 0.8 dk"
FOOD5K_SHUFFLED="C food5k_shuffled _tid_ 50 100 0.6 500 0.2 0.3 dk"
NYPD6="C nypd6 _ 324 100 0.9 60 0.05 0.3 dk"
SOCCER="C soccer _ 2000 100 0.9 40 0.05 0.3 dk"

DATASET=$HOSPITAL_SHUFFLED
LOGFILE=$(echo $DATASET | sed s/' '/'-'/g)

echo -n "source ../set_env.sh && "

for GRS in $GROUPING_STRATEGY; do
  for KLS in $KL_STRATEGY; do
    SCR="$DATASET"
    if [ $GRS == "pair_corr" ]; then
      GRT=$GROUPING_THRESH
    else
      if [ $GRS == "corr_sim" ]; then
        GRT=$SIM_THRESH
      else
        GRT="_"
      fi
    fi
    SCR="$SCR $GRS $GRT"
    if [ $KLS != "_" ]; then
      for KLT in $KL_THRESH; do
        SCR2="$SCR $KLS $KLT"
        LOGFILE=$(echo $SCR2 | sed s/' '/'-'/g)
        echo -n "$CLEANCMD && $PYCMD $SCR2 > $LOGFILE.log 2> $LOGFILE.err && "
      done
    else
      SCR="$SCR _ _"
      LOGFILE=$(echo $SCR | sed s/' '/'-'/g)
      echo -n "$CLEANCMD && $PYCMD $SCR > $LOGFILE.log 2> $LOGFILE.err && "
    fi
  done
done
    
echo "echo Finished!"

