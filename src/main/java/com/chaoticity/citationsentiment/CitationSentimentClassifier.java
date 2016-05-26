/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.chaoticity.citationsentiment;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.Random;

/**
 * Code and data for citation sentiment classification reported in http://www.aclweb.org/anthology/P11-3015
 * The file test.arff contains only the test set with dependency triplets generated with Stanford CoreNLP
 * Full corpus available at http://www.cl.cam.ac.uk/~aa496/citation-sentiment-corpus
 *
 * @author Awais Athar
 */
public class CitationSentimentClassifier {


    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("test.arff");
        Instances data = source.getDataSet();

        // Set class attribute
        data.setClassIndex(data.attribute("@@class@@").index());

        // delete unused attributes
        data.deleteAttributeAt(1);
        data.deleteAttributeAt(2);

        // split dependencies on space
        StringToWordVector unigramFilter = new StringToWordVector();
        unigramFilter.setInputFormat(data);
        unigramFilter.setIDFTransform(true);
        unigramFilter.setAttributeIndices("3");
        WordTokenizer whitespaceTokenizer = new WordTokenizer();
        whitespaceTokenizer.setDelimiters(" ");
        unigramFilter.setTokenizer(whitespaceTokenizer);
        data = Filter.useFilter(data,unigramFilter);

        // make trigrams from citation sentences
        StringToWordVector trigramFilter = new StringToWordVector();
        trigramFilter.setInputFormat(data);
        trigramFilter.setIDFTransform(true);
        trigramFilter.setAttributeIndices("2");
        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(1);
        tokenizer.setNGramMaxSize(3);
        trigramFilter.setTokenizer(tokenizer);
        data = Filter.useFilter(data,trigramFilter);

        // Train and test 10x cross-validation
        int folds = 10;
        LibSVM svm = new LibSVM();
        svm.setCost(1000);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(svm, data, folds, new Random(1));
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
    }


}
