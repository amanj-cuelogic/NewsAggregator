import sbt._

object Dependencies {
  lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.0.1"
  lazy val sparkCore = "org.apache.spark" % "spark-core_2.11" % "2.1.1" % "provided"
  lazy val sparkSQL = "org.apache.spark" % "spark-sql_2.11" % "2.1.1" % "provided"
  lazy val sparkML = "org.apache.spark" % "spark-mllib_2.11" % "2.1.1" % "provided"
}
