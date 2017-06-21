package example

import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{ Tokenizer, HashingTF, IDF, IDFModel }
import org.apache.spark.ml.clustering.{ KMeans, LDA }

import org.apache.log4j._
import org.apache.spark.ml.Pipeline

import opennlp.tools.tokenize.{ TokenizerModel, TokenizerME }
import opennlp.tools.util.Span
import java.io.FileInputStream
import java.io.File
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.namefind.NameFinderME
import opennlp.tools.util.InputStreamFactory

import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming._
import scala.collection.mutable.Map
import org.apache.spark.SparkConf
import org.apache.spark.sql.cassandra._

import com.datastax.spark.connector._
import com.datastax.spark.connector.cql._


object NewsAggregatorKMeans {

  def getModelDir = () => {
    "/home/aman/Downloads"
  }

  def main(args: Array[String]) = {

    val conf = new SparkConf()
      .setAppName("NewsAggregatorKafka")
      .setMaster("local[*]")
      .set("spark.cassandra.connection.host", "localhost")
    
    val ssc = new StreamingContext(conf, Seconds(60))

    Logger.getLogger("org").setLevel(Level.ERROR)

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> "localhost:9092",
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "newsAggregator",
      "connection.max.idle.ms" -> "54000",
      "retry.backoff.ms" -> "10000")

    val tweets = KafkaUtils
      .createDirectStream[String, String](
        ssc,
        PreferConsistent,
        Subscribe[String, String](
          Array("testTopic"),
          kafkaParams))

    val tweetsDS = tweets.map(record => (record.value()))

    val tokenStream = new FileInputStream(new File(getModelDir(), "en-token.bin"))
    val modelStream = new FileInputStream(new File("/home/aman/Downloads/en-ner-person.bin"))
    val tokenModel = new TokenizerModel(tokenStream)
    val tokenizernlp = new TokenizerME(tokenModel)
    val entityModel = new TokenNameFinderModel(modelStream)
    val nameFind = new NameFinderME(entityModel)

    val findName: (String => String) = (arg: String) => {

      val tokens = tokenizernlp.tokenize(arg.toString())
      val nameSpans = nameFind.find(tokens)
      Span.spansToStrings(nameSpans, tokens).mkString(",")

    }

    val nameFinder = udf(findName)

    tweetsDS.foreachRDD(rdd => {

      val spark = SparkSession.builder().config(rdd.sparkContext.getConf).getOrCreate()

      import spark.implicits._

      
      val tweets = rdd.collect().zipWithIndex
     
      val tweetsDF = spark.createDataFrame(tweets).toDF("sentence","id")
      
      val nerinputDF = tweetsDF.withColumn("ner", nameFinder('sentence))
      
      nerinputDF.select("ner").distinct().show(false)
      val kCount = nerinputDF.select("ner").distinct().count()
      
      val tokenizer = new Tokenizer()
        .setInputCol("ner")
        .setOutputCol("words")
      
      val hashingTF = new HashingTF()
        .setInputCol(tokenizer.getOutputCol)
        .setOutputCol("rawFeatures")

      val idf = new IDF()
        .setInputCol(hashingTF.getOutputCol)
        .setOutputCol("features")

      val pipeline = new Pipeline()
        .setStages(Array(tokenizer, hashingTF, idf))

      val idfmodel = pipeline.fit(nerinputDF)

      val featurizedData = idfmodel.transform(nerinputDF)

      val kmeans = new KMeans().setK(kCount.toInt).setSeed(1L)
      val kmodel = kmeans.fit(featurizedData)
      val test = kmodel.transform(featurizedData)

      test.select("sentence", "ner", "prediction").show(false)
      
      test.select("sentence", "ner", "prediction").write
        .mode("overwrite")
        .format("org.apache.spark.sql.cassandra")
        .options(Map("table"->"nerclusters","keyspace" -> "news"))
        .save()

    })

    ssc.start()
    ssc.awaitTermination()
  }
}