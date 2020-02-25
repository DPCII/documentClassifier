/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package nlp;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.LabelSeeker;
import tools.MeansBuilder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ParagraphVectorsClassifier {

    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsClassifier.class);

    public static String dataLocalPath = "/Users/imac/IdeaProjects/documentClassifier/src/main/java/data";


    public static void main(String[] args) throws Exception {

        ParagraphVectorsClassifier app = new ParagraphVectorsClassifier();

        // This is our vector builder
        app.makeParagraphVectors();

        // This is our vector consumer to perform categorization
        app.checkUnlabeledData();

    }


    // This builds vectors off of named folders containing files of any name, which serve as labels
    // (ie: /diplomacy which contains 1.txt, 2.txt, 3.txt, each filled with lines or paragraphs that you want the model
    // to learn as meaning "diplomacy")

    void makeParagraphVectors() throws Exception {
        File resource = new File(dataLocalPath, "paravec/labeled");

        List<String> removewords = Arrays.asList("!", ".", ",", "ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than");


        // build a iterator for our dataset
        iterator = new FileLabelAwareIterator.Builder()
            .addSourceFolder(resource)
            .build();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
            .stopWords(removewords)
            .minWordFrequency(2)
            .windowSize(5)
            .learningRate(0.025)
            .minLearningRate(0.001)
            .batchSize(1000)
            .epochs(250)
            .iterate(iterator)
//            .trainWordVectors(true)
            .tokenizerFactory(tokenizerFactory)
            .build();

        // Start model training
        paragraphVectors.fit();

        WordVectorSerializer.writeParagraphVectors(paragraphVectors, dataLocalPath + "paraVectors.pv");
        System.out.println("Serialized data is saved in " + dataLocalPath);

    }



    // This method performs categorization

    void checkUnlabeledData() throws IOException, IllegalStateException {

        try {
      /*
      At this point we assume that we have model built and we can check
      which categories our unlabeled document falls into.
      So we'll start loading our unlabeled documents and checking them
     */
            File unClassifiedResource = new File(dataLocalPath, "paravec/unlabeled");
            FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(unClassifiedResource)
                    .build();

     /*
      Now we'll iterate over unlabeled data, and check which label it could be assigned to
      Please note: for many domains it's normal to have 1 document fall into few labels at once,
      with different "weight" for each.
     */
            MeansBuilder meansBuilder = new MeansBuilder(
                    (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
                    tokenizerFactory);
            LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                    (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

            while (unClassifiedIterator.hasNextDocument()) {
                LabelledDocument document = unClassifiedIterator.nextDocument();
                INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
                List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

         /*
          please note, document.getLabel() is used just to show which document we're looking at now,
          as a substitute for printing out the whole document name.
          So, labels on these two documents are used like titles,
          just to visualize our classification done properly
         */
                log.info("Document '" + document.getLabels() + "' falls into the following categories: ");
                for (Pair<String, Double> score : scores) {
                    log.info("        " + score.getFirst() + ": " + score.getSecond());
                }
            }
        }
        catch(IllegalStateException e) {
            System.out.println(e);
        }

    }
}
