import static org.apache.spark.sql.functions.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

/**
 * https://spark.apache.org/docs/latest/api/java/index.html
 */
public class WikipediaPageRankWithoutThirdParty {
	public static void main(String[] args) {
		SparkSession spark = SparkSession
			.builder()
			.appName("PageRank")
			//.config("spark.master", "local") // only for local debug
			.getOrCreate();

		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
		jsc.hadoopConfiguration().set("textinputformat.record.delimiter", "<page>\n"); // config will be reused for all.
		jsc.setLogLevel("ERROR");
		
		spark.sparkContext().setLogLevel("ERROR");
		spark.sparkContext().setCheckpointDir("checkpoint");

		StructType linkSchema = new StructType()
				.add("source", DataTypes.StringType)
				.add("target", DataTypes.StringType)
				.add("redirectTo", DataTypes.StringType);
		
		Pattern titlePattern = Pattern.compile("<title>(.*?)</title>");
		Pattern textTagPattern = Pattern.compile("<text.*?</text>", Pattern.DOTALL);
		Pattern wikilinksPattern = Pattern.compile("\\[\\[(.*?)[\\|#\\]].*?\\]?\\]");
		Pattern redirectPattern = Pattern.compile("<redirect title=\\\"(.*?)\\\".*?\\/>");
		
		long start = System.currentTimeMillis();
		// read XML file delimted by "<page>\n"
		Dataset<String> xmlSplitted = spark.sqlContext().createDataset(jsc.textFile("wikipedia.xml").rdd(), Encoders.STRING());
		//Dataset<String> xmlSplitted = spark.sqlContext().createDataset(jsc.textFile("dewiki-latest-pages-articles.xml").rdd(), Encoders.STRING());
		
		// filter and parse relevant data
		Dataset<Row> links = xmlSplitted
				.flatMap((FlatMapFunction<String, Row>) s -> {
					// filter out
					if (!s.contains("<title>") || !s.contains("<ns>0</ns>")) {
						return Collections.emptyIterator();
					}

					Matcher titleMatcher = titlePattern.matcher(s);
					if(!titleMatcher.find()) {
						return Collections.emptyIterator();
					}
					String page = titleMatcher.group(1);
					
					// is redirect page
					String redirectTo = null;
					Matcher redirectMatcher = redirectPattern.matcher(s);
					if(redirectMatcher.find()) {
						redirectTo = redirectMatcher.group(1);
					}
					
					// find all linkings to other pages
					List<Row> linksTo = new ArrayList<>();
					Matcher textTagMatcher = textTagPattern.matcher(s);
					textTagMatcher.find();
					String textTag = textTagMatcher.group();
					if(textTag != null) {
						Matcher wikilinksMatcher = wikilinksPattern.matcher(textTag);
						while(wikilinksMatcher.find()) {
							linksTo.add(RowFactory.create(page, wikilinksMatcher.group(1), redirectTo));
						}
					}
					if(linksTo.isEmpty()) {
						// add the actual page which links to no one
						linksTo.add(RowFactory.create(page, null, redirectTo));
					}
					
					return linksTo.iterator();
				}, RowEncoder.apply(linkSchema));
		
		// Weiterleitungen auflösen
		Dataset<Row> allRedirects = links.filter("redirectTo is not null");
		links = links.as("allLinks")
			.join(allRedirects.as("allRedirects"), col("allLinks.target").equalTo(col("allRedirects.source")),"left")
			.filter("allLinks.redirectTo is null")
			.map((MapFunction<Row,Row>) row -> {
				String redirectTo = row.getString(5);
				if(redirectTo != null) {
					return RowFactory.create(row.getString(0), redirectTo, null);
				}
				else {
					return RowFactory.create(row.getString(0), row.getString(1), null);
				}
			}, RowEncoder.apply(linkSchema));
		
		
		// Nicht existente Seiten filtern. Subquery mit not exist gibt es nicht. 
		// Daher werden alle Daten behalten die auf existierende Sourcen oder NULL zeigen. (Semi Join + UnionAll)
		links = links.as("allLinks")
			.join(links.as("allSources"), col("allSources.source").equalTo(col("allLinks.target")), "semi")
			.unionAll(links.where("target IS NULL"))
			.select("source", "target"); // keep only relevant columns

		// PageRank
		final double d = 0.85;
		
		Dataset<Row> pages = links.select(col("source").as("page"))
			.union(links.select("target"))
			.distinct()
			.withColumn("rank", lit(1.0))
			.cache();

		links = links.groupBy("source")
			.agg(collect_list("target"))
			.cache();

		StructType resultSchema = new StructType()
			.add("target", DataTypes.StringType)
			.add("rank", DataTypes.DoubleType);

		System.out.println("pre pagerank took : " + (System.currentTimeMillis() - start) + " ms\n");
		for(int i = 0; i < 30; i++) {
			start = System.currentTimeMillis();
			Dataset<Row> old_pages = pages;

			Dataset<Row> contrib = links.join(pages, col("source").equalTo(col("page")))
				.flatMap((FlatMapFunction<Row, Row>) row -> {
					List<String> targets = row.<String>getList(1);
					double rank = row.<Double>getAs("rank") / targets.size();

					return targets.stream()
						.map(s -> RowFactory.create(s, rank))
						.collect(Collectors.toList())
						.iterator();
				}, RowEncoder.apply(resultSchema))
				.groupBy("target")
				.sum("rank");

			pages = pages.join(contrib, col("page").equalTo(col("target")), "left_outer")
				.select(col("page"), when(col("sum(rank)").isNotNull(), col("sum(rank)")).otherwise(0).multiply(d).plus(1.0 - d).as("rank"))
				.cache()
				.checkpoint();

			old_pages.unpersist();
			System.out.println("took: " + (System.currentTimeMillis() - start) + "ms\n");
		}

		pages.sort(desc("rank"))
			.write()
			.option("sep", "\t")
			.csv("result");
	}
}
