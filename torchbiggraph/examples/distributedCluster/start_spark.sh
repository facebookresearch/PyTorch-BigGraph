EXAMPLES_PATH="...../torchbiggraph/examples" # set correct path of pbg codebase

# compile the scala code and update below variables
JAR="" # set jar filename
CLASS_FILE="" # set scala class file

ZIP_FLDR='torchbiggraph'
TRAIN_WRAPPER='train_wrapper.py'
PBG_CONFIG="distributedCluster_config.py"
NUM_MACHINES=10

# create a zip file and send it cluster as part of spark submit
cwd=`pwd`
cd ${EXAMPLES_PATH}/../..
zip -r ${ZIP_FLDR}.zip ${ZIP_FLDR}

# start spark
spark-submit --class ${CLASS_FILE} \
  --master yarn \
  --deploy-mode cluster \
  --queue entgraph \
  --driver-memory=25G \
  --executor-memory=25G \
  --conf spark.executor.memoryOverhead=3G \
  --conf spark.driver.memoryOverhead=3G \
  --conf spark.driver.cores=10 \
  --conf spark.executor.cores=9 \
  --conf spark.task.cpus=9 \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.task.maxFailures=1 \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.yarn.nodemanager.vmem-check-enabled=false \
  --conf spark.locality.wait=0s \
  --conf spark.speculation=false \
  --conf spark.executorEnv.SPARK_YARN_USER_ENV=PYTHONHASHSEED=0 \
  --conf spark.yarn.appMasterEnv.ZIP_FILE=${ZIP_FLDR} \
  --conf spark.executorEnv.ZIP_FILE=${ZIP_FLDR} \
  --conf spark.yarn.appMasterEnv.TRAIN_WRAPPER=${TRAIN_WRAPPER} \
  --conf spark.executorEnv.TRAIN_WRAPPER=${TRAIN_WRAPPER} \
  --conf spark.yarn.appMasterEnv.PBG_CONFIG=${PBG_CONFIG} \
  --conf spark.executorEnv.PBG_CONFIG=${PBG_CONFIG} \
  --conf spark.yarn.appMasterEnv.NUM_MACHINES=${NUM_MACHINES} \
  --conf spark.executorEnv.NUM_MACHINES=${NUM_MACHINES} \
  --files ${ZIP_FLDR}.zip,${EXAMPLES_PATH}/distributedCluster/${TRAIN_WRAPPER},${EXAMPLES_PATH}/configs/${PBG_CONFIG} \
  ${JAR}

cd $cwd
