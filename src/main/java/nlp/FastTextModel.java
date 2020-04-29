package nlp;

import org.deeplearning4j.models.fasttext.FastText;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FastTextModel {

    FastText fastText;
    TokenizerFactory tokenizerFactory;

    private static final Logger log = LoggerFactory.getLogger(FastTextModel.class);

    public static void main(String[] args) {
        FastTextModel fastTextModel = new FastTextModel();

        fastTextModel.assessWordVectors();

    }

    public void assessWordVectors() {
//        File file = new File("/Users/imac/Downloads/sogou_news.bin");
        fastText = new FastText();

        fastText.loadBinaryModel("/Users/imac/Downloads/sogou_news.bin");


        System.out.println(fastText.hasWord("America"));

//        Collection<String> lst = fastText.wordsNearestSum("day", 10);
//
//        for(String str : lst)
//            System.out.println(str);
    }
}
