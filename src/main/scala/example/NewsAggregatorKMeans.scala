package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{ Tokenizer, HashingTF, IDF, IDFModel }
import org.apache.spark.ml.clustering.{KMeans, LDA}

import org.apache.log4j._
import org.apache.spark.ml.Pipeline

import opennlp.tools.tokenize.{TokenizerModel, TokenizerME}
import opennlp.tools.util.Span
import org.apache.spark.sql.Row
import java.io.FileInputStream
import java.io.File
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.namefind.NameFinderME
import scala.collection.mutable.ArrayBuffer
import java.io.BufferedOutputStream
import java.io.FileOutputStream
import opennlp.tools.util.InputStreamFactory
import opennlp.tools.util.PlainTextByLineStream
import opennlp.tools.namefind.NameSampleDataStream
import java.util.Collections
import opennlp.tools.util.TrainingParameters
import opennlp.tools.namefind.TokenNameFinderFactory


object NewsAggregatorKMeans {
  
  def getModelDir = () => {
    "/home/aman/Downloads"
  }

  def main(args: Array[String]) = {

    val spark = SparkSession.builder()
      .appName("NewsAggregatorKMeans")
      .master("local[*]")
      .getOrCreate()

    Logger.getLogger("org").setLevel(Level.ERROR)
      
    val inputDF = spark.createDataFrame(Seq(
      (0, "You can watch all the Lucknow programmes of PM @narendramodi live on your mobiles.", 1.0),
      (1, "Tomorrow morning, on 3rd #YogaDay, PM @narendramodi will join the Yoga Day programme at the Ramabai Ambedkar Maidan in Lucknow.", 1.0),
      (2, "PM @narendramodi will distribute sanction letters to beneficiaries of PM Awas Yojana. ‘Housing for all’ is a key priority for NDA Govt.", 1.0),
      (3, "We will support Shri Ramnath Kovind for President of India. PM @narendramodi spoke to me about his candidature and sought our support 1/2", 1.0),
      (4, "\"Yoga has spread harmony between man and nature. It is a holistic approach\"-Narendra Modi", 2.0),
      (5, "Narendra Modi says Shimla election victory a reflection of people's faith in 'development politics' http://crwd.fr/2rKKE9n ", 2.0),
      (6, "Donald Trump has been quick to condemn Islamist terror. He's still silent on the London mosque attack http://ind.pn/2rKU4lo ", 2.0),
      (7, "my heart goes out to the ppl who think donald trump is actually a good president. i feel so bad that you're that fucking stupid.", 2.0),
      (8, "OBAMA: Donald Trump is trying to destroy my legacy AMERICA: That's why we VOTED for #Trump", 0.0),
      (9, "This woman explains why she left the Democratic Party and Voted for Donald Trump #DemExit #MAGA", 0.0),
      (10, "Tim Cook's face at Donald Trump's tech roundtable is everything http://on.mash.to/2sJsB2r ", 0.0),
      (11, "Donald Trump is Motivated By his  HATE of Barack Obama & Love for Money, Not His LOVE for America.", 3.0),
      (12, "On \"The Late Show,\" Stephen Colbert and Seth Rogen read some direct messages they sent Donald Trump Jr. on Twitter http://cnn.it/2slrgP3 ", 3.0),
      (13, "PM @narendramodi will also inaugurate a building of the Dr. A.P.J. Abdul Kalam Technical University.", 3.0),
      (14, "In Lucknow, PM @narendramodi will dedicate to the nation the 400 KV Lucknow-Kanpur D/C transmission line.", 3.0)))
      .toDF("id", "sentence", "group")

    
    import spark.implicits._
    
//    val trainingParam = new TrainingParameters()
//    trainingParam.put("iterations", 100)
//    trainingParam.put("cutoff",5)
//    
//    val modelOutputStream = new BufferedOutputStream(
//      new FileOutputStream(new File("/home/aman/Downloads/demo/modelFile"))    
//    )
//    
//    val isf = new InputStreamFactory() {
//      
//      def createInputStream() = {
//        new FileInputStream("/home/aman/Downloads/en-ner-person.train")
//      }
//    }
//    
//    val lineStream = new PlainTextByLineStream(isf, "UTF-8")
//    
//    val sampleStream = new NameSampleDataStream(lineStream)
//    
//    val opennlpmodel = NameFinderME.train("en","person",sampleStream,trainingParam,new TokenNameFinderFactory())
    
//    opennlpmodel.serialize(modelOutputStream)
    
    val tokenStream = new FileInputStream(new File(getModelDir(), "en-token.bin"))
    val modelStream = new FileInputStream(new File("/home/aman/Downloads/en-ner-person.bin"))
    val tokenModel = new TokenizerModel(tokenStream)
    val tokenizernlp = new TokenizerME(tokenModel)
    val entityModel = new TokenNameFinderModel(modelStream)
    val nameFind = new NameFinderME(entityModel)
    
    val findName : (String => Array[String]) = (arg: String) => { 
  
      val tokens = tokenizernlp.tokenize(arg)
      val nameSpans = nameFind.find(tokens)
      val length = nameSpans.length
      val ners = new ArrayBuffer[String]()
      Span.spansToStrings(nameSpans, tokens)
      
    }
    
    val nameFinder = udf(findName)
    
    val nerinputDF = inputDF.withColumn("ner", nameFinder('sentence))
    
    nerinputDF.select("ner").distinct().show(false)
   
    
//    val tokenizer = new Tokenizer()
//      .setInputCol("sentence")
//      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setInputCol("ner")
      .setOutputCol("rawFeatures")

    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(hashingTF, idf))
      
    val idfmodel = pipeline.fit(nerinputDF)
    
    val featurizedData = idfmodel.transform(nerinputDF)
//    
//    //val Array(trainingData, testData) = featurizedData.randomSplit(Array(0.7,0.3),seed = 12345L)
//    
    //featurizedData.show(20)
    
    val kmeans = new KMeans().setK(5).setSeed(1L)
    val kmodel = kmeans.fit(featurizedData)
    val test =  kmodel.transform(featurizedData)
    
    test.select("sentence","ner","prediction").show(false)
    
      
  }
}