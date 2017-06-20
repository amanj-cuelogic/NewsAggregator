import sbt._

object Dependencies {
  lazy val commonBeanUtils = "commons-beanutils" % "commons-beanutils" % "1.9.3"
  lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.0.1"
  lazy val sparkCore = "org.apache.spark" % "spark-core_2.11" % "2.1.1" % "provided"
  lazy val sparkSQL = "org.apache.spark" % "spark-sql_2.11" % "2.1.1" % "provided"
  lazy val sparkML = "org.apache.spark" % "spark-mllib_2.11" % "2.1.1" % "provided"
  lazy val sparkStreaming = "org.apache.spark" % "spark-streaming_2.11" % "2.1.1" % "provided"
  lazy val sparkKafka = "org.apache.spark" % "spark-sql-kafka-0-10_2.11" % "2.1.1" % "provided"
  lazy val sparkStreamingKafka = "org.apache.spark" % "spark-streaming-kafka-0-10_2.11" % "2.1.1"
  lazy val cassandraConnector = "com.datastax.spark" % "spark-cassandra-connector_2.11" % "2.0.2"
  lazy val kafkaClients = "org.apache.kafka" % "kafka-clients" % "0.10.2.1"
  lazy val opennlp = "org.apache.opennlp" % "opennlp-tools" % "1.8.0"
}
