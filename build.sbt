import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.11.7",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "Hello",
    libraryDependencies ++= Seq(
    	commonBeanUtils,
	    scalaTest % Test,
	    sparkCore,
      	sparkSQL,
      	sparkML,
      	sparkStreaming,
      	sparkKafka,
      	cassandraConnector,
      	kafkaClients,
      	sparkStreamingKafka
	  )
  )
