package nlp;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.util.Collection;

public class Word2VecModel {

    private static final String corpusPath = "/Users/imac/Downloads/wikitext-103/wiki.train.tokens";

    private static Logger log = LoggerFactory.getLogger(Word2VecModel.class);

    public static void main(String[] args) throws FileNotFoundException {

        log.info("Get corpus, extract each line, tokenize each word");
        SentenceIterator sentenceIterator = new BasicLineIterator(corpusPath);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Start building Word2Vec model...");
        Word2Vec model = new Word2Vec.Builder()
                .minWordFrequency(2)
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizerFactory)
                .batchSize(500)
                .iterations(1)
                .epochs(5)
                .seed(42)
                /* Window size is how many words before and after the current word gets included in analysis (includes
                current word - should always be odd) */
                .windowSize(5)
                .build();

        log.info("Model constructed. Now fitting the model...");
        model.fit();

        log.info("Writing word vectors...");
//        ModelSerializer.
        WordVectorSerializer.writeWord2VecModel(model, "/Users/imac/IdeaProjects/documentClassifier/src/w2vModel1");

        log.info("Testing result: What are the closest words to: America");
        Collection<String> lst = model.wordsNearestSum("America", 10);
        log.info("{}", lst);






    }

}
