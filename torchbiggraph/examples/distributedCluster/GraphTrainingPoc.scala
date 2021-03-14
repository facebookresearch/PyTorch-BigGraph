// compile this scala code and use to run spark submit job

import java.net._
import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.sys.process.Process

object GraphTrainingPoc  {

    val ZIP_FILE: String = sys.env("ZIP_FILE") //'torchbiggraph'
    val TRAIN_WRAPPER: String = sys.env("TRAIN_WRAPPER") // "train_wrapper.py"
    val PBG_CONFIG: String = sys.env("PBG_CONFIG") // "distributedCluster_config.py"
    val NUM_MACHINES: Int = sys.env("NUM_MACHINES").toInt

    def main(args: Array[String]) {

        val num_machines = NUM_MACHINES

        val spark = SparkSession.builder().enableHiveSupport().getOrCreate()
        implicit val sparkContext: SparkContext = spark.sparkContext

        //Step 1: Get host and port of Driver
        val (driverHost: String, driverPort: Int) = getHostAndPort

        //Step 2: Ask resource manager for required number of executors to run all workers
        requestExecutorsAndWait(num_machines)

        //Step 3:  execute PBG process on driver
        // Driver will wait until all trainer not complete process group
        val driverFuture = Future {
            executePbgProcess(0, driverHost, driverPort)
        }

        //Step 4: execute PBG process on each worker trainers i.e on each partition
        sparkContext
            .makeRDD((1 until num_machines).toList, 1)
            .repartition(num_machines - 1)
            .mapPartitionsWithIndex(
                (index: Int, iterator: Iterator[Int]) => {
                    executePbgProcess(index + 1, driverHost, driverPort)
                    iterator
                }
            ).collect()

        Await.result(driverFuture, Duration.Inf)
    }

    def extractZip(targetPath: String, zipName: String, logPrefix: Option[String] = None): Boolean = {

        val prefix = logPrefix.getOrElse("")

        val containerPath = Paths.get(".").toAbsolutePath.toString
        print(s"${prefix}ContainerPath - $containerPath")
        print(s"${prefix}Check if path exists - $targetPath/$zipName")

        val pathAlreadyFound = Files.exists(Paths.get(s"$targetPath/$zipName"))

        // unzip contents if path doesnt exists
        if (pathAlreadyFound) {
            print(s"${prefix}Folder $zipName already exists. Skip unzip..")
        }
        else {
            print(s"${prefix}Folder $zipName not found. Unzip $zipName.zip")

            val pzip = Process(s"unzip $targetPath/$zipName.zip -d $targetPath")
            if (pzip.! != 0) {
                val errorMsg = s"${prefix}Error doing unzip of $zipName.zip"
                print(errorMsg)
                throw new Exception(errorMsg)
            }

            print(s"${prefix}Unzip $zipName successful..")
        }

        true
    }

    /**
      * This function is used to get Host and port of Trainer 0 which will be executed on driver
      */
    def getHostAndPort: (String, Int) = {
        // get ip address and port of the machine
        val localhost: String = InetAddress.getLocalHost.getHostAddress
        var port = -1
        try {
            val s = new ServerSocket(0)
            port = s.getLocalPort
            s.close()
        }
        catch {
            case e: Exception => print(e.getMessage)
        }

        (localhost, port)
    }

    /**
      * This function is used get executor from resource manager
      */
    def requestExecutorsAndWait(numExecutors: Int)(implicit sparkContext: SparkContext): Boolean = {

        val workerExecCount = numExecutors.toInt - 1
        if (sparkContext.requestTotalExecutors(workerExecCount, 0, Map.empty)) {

            var counter = 1

            while (sparkContext.getExecutorStorageStatus.length - 1 < workerExecCount) {
                print(s"Waiting for $workerExecCount executors. Iter - $counter")
                Thread.sleep(60000)
                counter += 1
            }

            if (counter > 15) {
                val errorMsg = s"Unable to get the required executors in 15 minutes " +
                    s"- Expected: $workerExecCount, Got: ${sparkContext.getExecutorStorageStatus.length}"
                print(errorMsg)
                throw new Exception(errorMsg)
            }
        } else {

            return false
        }

        print(s"Got requested executors. Count - $workerExecCount")
        true
    }

    def executePbgProcess(rank: Int, driverHost: String, driverPort: Int): Int = {

        val executionPath = Paths.get(".").toAbsolutePath.toString

        // Setup driver machine, remove PBG zip code if exists and place fresh copy
        extractZip(zipName = ZIP_FILE, targetPath = executionPath, logPrefix = Some(s"[Rank $rank] : "))

        val command = "python3.7 " + executionPath + "/" + TRAIN_WRAPPER + " --rank  " + rank.toString + " " + PBG_CONFIG

        print(s"[Rank $rank] : command: " + command)
        val rCode = runCommand(command = command,
            path = executionPath,
            extraEnv = List("MASTER_ADDR" -> driverHost,
                "MASTER_PORT" -> driverPort.toString,
                "WORLD_SIZE" -> NUM_MACHINES.toString,
                "PYTHONPATH" -> ".")
        )
        rCode
    }

    def runCommand(command: String, path: String, extraEnv: List[(String, String)] = List.empty): Int = {
        val ps = Process(command = Seq("/bin/sh", "-c", command),
            cwd = Paths.get(path).toFile,
            extraEnv = extraEnv: _*
        )

        val proc = ps.run()
        val exitValue = proc.exitValue()

        if (exitValue != 0)
            throw new RuntimeException(s"Process exited with code $exitValue")

        exitValue

    }
}