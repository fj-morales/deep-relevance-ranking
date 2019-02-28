import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.index.stats.NodeStatistics;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;

import java.util.Set;
import java.util.TreeSet;

/**
 * The is an example for transforming between internal and external IDs in Galago.
 *
 * @author Jiepu Jiang (jpjiang@cs.umass.edu)
 * @version 2016-09-10
 */
public class extractStatistics {

public static void main( String[] args ) {
	try {

		String pathIndexBase = "./baseline_files/index_bioasq_test/";

		// Let's just count the IDF and P(w|corpus) for the word "reformulation" in the "text" field
		String field = "TEXT";
		String term = "enzime";

		Retrieval retrieval = RetrievalFactory.instance( pathIndexBase );

		Node termNode = StructuredQuery.parse( "#text:" + term + ":part=field." + field + "()" );
		termNode.getNodeParameters().set( "queryType", "count" );

		NodeStatistics termStats = retrieval.getNodeStatistics( termNode );
		long corpusTF = termStats.nodeFrequency; // Get the total frequency of the term in the text field
		long n = termStats.nodeDocumentCount; // Get the document frequency (DF) of the term (only counting the text field)

		Node fieldNode = StructuredQuery.parse( "#lengths:" + field + ":part=lengths()" );
		FieldStatistics fieldStats = retrieval.getCollectionStatistics( fieldNode );
		long corpusLength = fieldStats.collectionLength; // Get the length of the corpus (only counting the text field)
		long N = fieldStats.documentCount; // Get the total number of documents

		double idf = Math.log( ( N + 1 ) / ( n + 1 ) ); // well, we normalize N and n by adding 1 to avoid n = 0
		double pwc = 1.0 * corpusTF / corpusLength;

		System.out.printf( "%-30sN=%-10dn=%-10dIDF=%-8.2f\n", term, N, n, idf );
		System.out.printf( "%-30slen(corpus)=%-10dfreq(%s)=%-10dP(%s|corpus)=%-10.6f\n", term, corpusLength, term, corpusTF, term, pwc );

		retrieval.close();

	} catch ( Exception e ) {
		e.printStackTrace();
	}
}

}
