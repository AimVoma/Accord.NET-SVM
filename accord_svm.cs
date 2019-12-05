using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.IO;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics;
using Accord.MachineLearning;
using System.IO;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using System.Net;
using System.Text.RegularExpressions;
using OpenNLP.Tools.SentenceDetect;
using Annytab.Stemmer;
using Accord.Statistics.Analysis;
using System.Diagnostics;
using Accord.MachineLearning.Performance;
using Accord.Math.Optimization.Losses;
using Word2vec.Tools;


/*
  Machine Learning Application that performs Sentiment Classification(Fine-Grained, emotions)
  with sparse text representations(TFIDF) or pre-trained dense word vectors(Word2Vec),
  on Supervised Linear Model(SVM). The Classification result is later dumped
  in local storage as Confusion Matrix.
*/

namespace SVM_MACHINE_LEARNING
{
    /*
      Support class for convenient dataset pre-process, extracting pairs
      (text-emotion-line) from source dataset.
    */
    public class pair
    {
        public int pair_line { get; set; }
        public string pair_text { get; set; }
        public string pair_emot { get; set; }
        public override string ToString()
        {
            return "Sentence: " + this.pair_text + "   Emotion: " + this.pair_emot + " LineNo:  " + this.pair_line;
        }
    }
    /*
      SVM Logic Execution of One-Vs-Rest Classifier. Text Representations are either sparse(TFIDF) or dense
      (Word2Vec) Vector Space Models(VSMs).
    */
    class svm_execute
    {
        public static string[] model = System.Configuration.ConfigurationManager.AppSettings["model"].Split(',');
        public static string output_location = System.Configuration.ConfigurationManager.AppSettings["output_location"];
        private static List<string> Missing_Tokens = new List<string>();


        static void Main(string[] args)
        {
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            svm_classifier Classifier = new svm_classifier();
            List<ConfusionMatrix> conf_matrix = new List<ConfusionMatrix>();
            string training_set = System.Configuration.ConfigurationManager.AppSettings["training_set"];
            string testing_set = System.Configuration.ConfigurationManager.AppSettings["testing_set"];

            switch (System.Configuration.ConfigurationManager.AppSettings["indexing_method"])
            {
                case "TFIDF":
                        TF_IDF Indexation = new TF_IDF();
                        Classifier.init_trainSamples(Indexation.Calculate("train_data"));
                        Classifier.init_testSamples(Indexation.Calculate("test_data"));
                        break;
                case "W2V":
                        string w2v_preTrained = System.Configuration.ConfigurationManager.AppSettings["w2v_preTrained"];
                        W2V.load(w2v_preTrained);
                        Classifier.init_trainSamples(W2V.Vectorize(W2V.Extract_sentences("train_data")));
                        Classifier.init_testSamples(W2V.Vectorize(W2V.Extract_sentences("test_data")));
                    break;
            }

            Console.WriteLine("Training Data: " + Classifier.get_trainData().Count());
            Console.WriteLine("Testing Data: " + Classifier.get_testData().Count());

            Console.WriteLine("Begin Training! -- Trainset: " + training_set);
            Console.WriteLine("_______________");

            foreach (string _emotion in model)
                Classifier.train(svm_preprocess.create_pairs("train_data"), _emotion);

            Console.WriteLine("Begin Testing! -- Testset " + testing_set);
            Console.WriteLine("_______________");

            conf_matrix.Clear();
            foreach (string _emotion in model)
                conf_matrix.Add(Classifier.test(svm_preprocess.create_pairs("test_data"), _emotion));

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);

            Console.WriteLine("Serializing!");
            Console.WriteLine("_______________");

            ToFile.Dump(Path.Combine(output_location, "corpus[Train " + training_set +
                                    " -- " + "Test " + testing_set + "]" + ".txt"), conf_matrix, model, elapsedTime);
        }
    }
    // Sentiment Dataset pre-process, IO operations, Text Preprocess
    static class svm_preprocess
    {
        internal static List<pair> create_pairs(string data_type)
        {
            string dataset_location = System.Configuration.ConfigurationManager.AppSettings["dataset_location"];
            string training_set = System.Configuration.ConfigurationManager.AppSettings["training_set"];
            string testing_set = System.Configuration.ConfigurationManager.AppSettings["testing_set"];
            string d_index = System.Configuration.ConfigurationManager.AppSettings["delimeter"];
            string training_folder = training_set + "/"; string testing_folder = testing_set + "/";
            Dictionary<string, char> Delimeter = new Dictionary<string, char>()
            {
                {"alpha", '@'},
                {"tab", '\t'},
                {"whitespace", ' '}
            };

            List<pair> tmp_pairs = new List<pair>();
            string file = "";

            switch (data_type)
            {
                case "IDF_Dictionary":
                    file = dataset_location + training_folder + training_set + "[TFIDF]" + ".txt";
                    break;
                case "train_data":
                    file = dataset_location + training_folder + training_set + "[train]" + ".txt";
                    break;
                case "test_data":
                    file = dataset_location + testing_folder + testing_set + "[test]" + ".txt";
                    break;
                default:
                    Console.WriteLine("Unavailable Dataset Location");
                    file = "FileNotFound";
                    break;
            }
            /*
                Read The File and Extract the Emotion, Text Fields and File Line Fields
            */
            int line_counter = 0;
            string delimeter = System.Configuration.ConfigurationManager.AppSettings["delimeter"];
            string emotion = System.Configuration.ConfigurationManager.AppSettings["emotion"];
            string sentence = System.Configuration.ConfigurationManager.AppSettings["sentence"];

            foreach (string line in File.ReadLines(@file))
            {
                ++line_counter;
                tmp_pairs.Add(new pair()
                {
                    pair_emot = line.Split(Delimeter[d_index])[Int32.Parse(emotion)],
                    pair_text = line.Split(Delimeter[d_index])[Int32.Parse(sentence)],
                    pair_line = line_counter
                });
            }
            return tmp_pairs;
        }

        /*
        We use ReadAllLines method for the reason that the dataset we load on memory is countable/static
        */
        internal static string[] Get_StopW()
        {
            return File.ReadAllLines(@"../../Dataset/StopWords_Filter.txt", Encoding.UTF8).ToArray();
        }

        internal static List<pair> PreProcess(List<pair> pair_list)
        {
            /*
            Pre-processing Steps:
                #) Apply Stop Words Removal
                #) For non Sto-Words Strings perform Stemming and concatenate them into single String
                #) Recycle the old entry
            */
            int index_counter = 0;
            List<string> str_dump = new List<string>();
            IStemmer stemmer = new EnglishStemmer();
            foreach (pair item in pair_list)
            {
                var Stop_Words = Get_StopW();
                str_dump.Clear();
                foreach (string word in item.pair_text.Split())
                {
                    if (Stop_Words.Contains(word.ToLower()))
                        continue;
                    else
                    {
                        if (word.Length > 2)
                        {
                            if (System.Configuration.ConfigurationManager.AppSettings["indexing_method"] == "W2V")
                                str_dump.Add(word.ToLower());
                            else
                                str_dump.Add(stemmer.GetSteamWord(word.ToLower()));
                        }
                        else
                            continue;
                    }
                }
                // Concatenation & Recycling
                pair_list[index_counter].pair_text = str_dump.Aggregate((i, j) => i + ' ' + j);
                index_counter++;
            }
            return pair_list;
        }
    }

    /*
      Apply on demand different optimization functions(see S.M.O) on Linear
      classifier's training algorithm(SMO, LCD), train-test serialization
      and produce Confusion Matrix
    */
    class svm_classifier
    {
        public List<double[]> train_sample = new List<double[]>();
        public List<double[]> test_sample = new List<double[]>();
        string Filepath = "";
        List<int> svm_feed_test = new List<int>();
        List<int> svm_feed_train = new List<int>();
        string trained_location = System.Configuration.ConfigurationManager.AppSettings["trained_location"];
        string kernel = System.Configuration.ConfigurationManager.AppSettings["kernel"];
        SequentialMinimalOptimization<Linear> svm_learner_smo;
        LinearCoordinateDescent<Linear> svm_learner_lcd;
        dynamic learner;

        List<double[]> idf_weights = new List<double[]>();
        public svm_classifier() : base()
        {
            if (System.Configuration.ConfigurationManager.AppSettings["train_function"] == "SMO")
            {
                learner = new SequentialMinimalOptimization<Linear>()
                {
                    UseComplexityHeuristic = true
                };
            }
            else if (System.Configuration.ConfigurationManager.AppSettings["train_function"] == "LCD")
            {
                learner = new LinearCoordinateDescent<Linear>()
                {
                    UseComplexityHeuristic = true
                };
            }
        }

        public void init_trainSamples(List<double[]> train_set) {
            this.train_sample = train_set;
        }
        public void init_testSamples(List<double[]> test_set) {
            this.test_sample = test_set;
        }

        public void train(List<pair> train_data, string emotion)
        {
            if (Boolean.Parse(System.Configuration.ConfigurationManager.AppSettings["rep_results"]) == true)
                Accord.Math.Random.Generator.Seed = 0;

            this.svm_feed_train.Clear();
            for (int _counter = 0; _counter < train_data.Count; _counter++)
            {
                if (train_data[_counter].pair_emot == emotion)
                    this.svm_feed_train.Add(1);
                else
                    this.svm_feed_train.Add(0);
            }

            if (Boolean.Parse(System.Configuration.ConfigurationManager.AppSettings["use_weights"]) == true)
                learner.WeightRatio = Double.Parse(System.Configuration.ConfigurationManager.AppSettings["weight_ratio"]);

            SupportVectorMachine<Linear> svm = learner.Learn(this.train_sample.ToArray(),this.svm_feed_train.ToArray());
            this.Filepath = Path.Combine(trained_location, emotion + "__" + kernel);
            Serializer.Save(svm, this.Filepath);
        }

        public ConfusionMatrix test(List<pair> test_data, string emotion)
        {
            if (Boolean.Parse(System.Configuration.ConfigurationManager.AppSettings["rep_results"]) == true)
                Accord.Math.Random.Generator.Seed = 0;

            this.Filepath = Path.Combine(trained_location, emotion + "__" + kernel);
            SupportVectorMachine<Linear> svm = Serializer.Load<SupportVectorMachine<Linear>>(this.Filepath);

            bool[] prediction = svm.Decide(this.test_sample.ToArray());
            int[] results = prediction.ToZeroOne();

            this.svm_feed_test.Clear();
            for (int _counter = 0; _counter < test_data.Count; _counter++)
            {
                if (test_data[_counter].pair_emot == emotion)
                    this.svm_feed_test.Add(1);
                else
                    this.svm_feed_test.Add(0);
            }

            return (new ConfusionMatrix(results, this.svm_feed_test.ToArray(), 1, 0));
        }

        public List<double[]> get_trainData()
        {
            return this.train_sample;
        }

        public List<double[]> get_testData()
        {
            return this.test_sample;
        }
    }
    #region l2norm + ToFile
    public static class L2Norm
    {
        internal static double[] Normalize(double[] vector)
        {
            List<double> result = new List<double>();
            double sumSquared = 0;
            foreach (var value in vector)
                sumSquared += value * value;

            double SqrtSumSquared = Math.Sqrt(sumSquared);
            foreach (var value in vector)
            {
                // L2-norm: Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
                result.Add(value / SqrtSumSquared);
            }
            return result.ToArray();
        }
    }
    /*
      Dump To File, Confusion Matrix elements(EM, TP, TN, FN, PR, RE, F1)
    */
    public static class ToFile
    {
        internal static void Dump(string pathToFile, List<ConfusionMatrix> conf_matrix, string[] Emotions, string elapsedT)
        {
            int counter = 0;
            using (StreamWriter outputFile = new StreamWriter(pathToFile, false))
            {
                outputFile.Write("EM" + "\t");
                outputFile.Flush();
                outputFile.Write("TP" + "\t");
                outputFile.Flush();
                outputFile.Write("FP" + "\t");
                outputFile.Flush();
                outputFile.Write("TN" + "\t");
                outputFile.Flush();
                outputFile.Write("FN" + "\t");
                outputFile.Flush();
                outputFile.Write("PR" + "\t");
                outputFile.Flush();
                outputFile.Write("RE" + "\t");
                outputFile.Flush();
                outputFile.Write("F1" + "\t");
                outputFile.Flush();
                outputFile.WriteLine();
                foreach (ConfusionMatrix cm in conf_matrix)
                {
                    outputFile.Write(Emotions[counter++] + "\t");
                    outputFile.Flush();
                    outputFile.Write(cm.TruePositives + "\t");
                    outputFile.Flush();
                    outputFile.Write(cm.FalsePositives + "\t");
                    outputFile.Flush();
                    outputFile.Write(cm.TrueNegatives + "\t");
                    outputFile.Flush();
                    outputFile.Write(cm.FalseNegatives + "\t");
                    outputFile.Flush();
                    outputFile.Write(String.Format("{0:0.00}", cm.Precision) + "\t");
                    outputFile.Flush();
                    outputFile.Write(String.Format("{0:0.00}", cm.Recall) + "\t");
                    outputFile.Flush();
                    outputFile.Write(String.Format("{0:0.00}", cm.FScore) + "\t");
                    outputFile.Flush();

                    outputFile.WriteLine();
                }
                outputFile.Flush();
                outputFile.WriteLine("Runtime:__ " + elapsedT);
                outputFile.Close();
            }
        }

    }
    #endregion
    //Custom TFIDF Implementation
    #region TFIDF
    public class TF_IDF
    {
        TFIDF codebook;
        public TF_IDF()
        {
            string[][] sentences = { };
            string data_type = "IDF_Dictionary";
            // Initialize TFIDF
            this.codebook = new TFIDF()
            {
                Tf = TermFrequency.Log,
                Idf = InverseDocumentFrequency.Default
            };

            sentences = Extract_sentences(data_type);

            // TFIDF Document Train
            codebook.Learn(sentences);
        }

        public string[][] Extract_sentences(string data_type)
        {
            List<string> _text = new List<string>();

            foreach (pair tmp_pair in svm_preprocess.create_pairs(data_type))
                _text.Add(tmp_pair.pair_text);
            return (string[][])_text.ToArray<string>().Tokenize();
        }
        public List<double[]> Calculate(string dataset)
        {
            List<double[]> idf_weights = new List<double[]>();
            List<double> _preprocessing = new List<double>();
            string[][] sentences = { };

            sentences = Extract_sentences(dataset);

            int _counter = 0;
            idf_weights.Clear();
            do
            {
                _preprocessing.Clear();

                foreach (double _temp in codebook.Transform(sentences[_counter]))
                    _preprocessing.Add(_temp);

                idf_weights.Add(_preprocessing.ToArray<double>());
            } while (_counter++ < sentences.Length - 1);

            //L2 Normalization
            _counter = 0;
            do
            {
                idf_weights[_counter] = L2Norm.Normalize(idf_weights[_counter]);
            } while (_counter++ < idf_weights.Count - 1);

            return idf_weights;
        }
    }
    #endregion
    #region W2V
    //Word2Vec pre-trained word vector variant for Text Representation, instead of TFIDF
    public static class W2V
    {
        static List<string> Missing_Tokens = new List<string>();
        static Vocabulary w2v_vocabulary;
        static double mw_percentage, tokens, miss_tokens = 0.0;

        public static void load(string path) {
            w2v_vocabulary = new Word2VecTextReader().Read(path);
        }

        public static List<string[]> Extract_sentences(string data_type)
        {
            List<string[]> _text = new List<string[]>();

            foreach (pair tmp_pair in svm_preprocess.create_pairs(data_type))
                _text.Add(tmp_pair.pair_text.Split());

            return _text;
        }

        public static List<double[]> Vectorize(List<string[]> Document)
        {
            List<double[]> reslt = new List<double[]>();
            foreach (string[] sentence in Document)
            {
                List<double> sentence_vec = new List<double>();
                List<double[]> wrd_vec = new List<double[]>();

                foreach (string w in sentence)
                {
                    tokens++;
                    if (w2v_vocabulary.ContainsWord(w))
                        wrd_vec.Add(Array.ConvertAll<float,double>(w2v_vocabulary.GetRepresentationFor(w).NumericVector, x => (double)x));
                    else if (!Missing_Tokens.Contains(w))
                       Missing_Tokens.Add(w);
                    else
                        miss_tokens++;
                }
                for (int i = 0; i<w2v_vocabulary.VectorDimensionsCount; i++)
                {
                    if (wrd_vec.Count >= 1)
                    {
                        double avg = 0;
                        for (int j = 0; j < wrd_vec.Count; j++)
                        {
                            avg = (double)wrd_vec[j][i] + (double)avg;
                            //Averaging
                            if (j + 1 == wrd_vec.Count)
                                avg = (double)(avg / (double)wrd_vec.Count);
                        }
                        sentence_vec.Add(avg);
                    }
                    else
                    {
                        sentence_vec.Add((double)0);
                    }
                }
                reslt.Add(sentence_vec.ToArray());
            }
            return reslt;
        }
    }
    #endregion
}

//________________________________ CROSSVALIDATION ________________________________________
this.crossvalidation = new Accord.MachineLearning.Performance.CrossValidation<SupportVectorMachine<Linear, double[]>, double[]>()
// If needed, control the parallelization degree
this.crossvalidation.ParallelOptions.MaxDegreeOfParallelism = 1;
{
   // Cross-Validation Folds
   K = 10,

   // Indicate how learning algorithms for the models should be created
   Learner = (s) => new SequentialMinimalOptimization<Linear, double[]>()
   {
       Complexity = 100,
       UseKernelEstimation = true
   },

   // Indicate how the performance of those models will be measured
   Loss = (expected, actual, p) => new Accord.Math.Optimization.Losses.ZeroOneLoss(expected).Loss(actual),

   Stratify = false, // do not force balancing of classes
};

/*          _________________________________TFIDF-DEBUG___________________________________________

            int _counter_line = 0;
            int _counter_2 = 0;
            int _counter_zero = 0;
            int _counter_NewLine = 75;
            do
            {
                _counter_2 = 0;
                _counter_zero = 0;
                foreach (double idf_element in idf_weights[_counter_line++])
                {
                    if (idf_element == 0)
                        _counter_zero++;

                    if (_counter_2 % _counter_NewLine == 0)
                        Console.WriteLine();

                    Console.Write(" " + Math.Round(idf_element, 2));
                    _counter_2++;
                }
                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("Dict Elements: " + _counter_2);
                Console.WriteLine("Non-Zero Elements: " + _counter_zero);
                Console.WriteLine("Line No: " + _counter_line);
                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine();
            } while (_counter_line < _counter);

            Console.Read();
            Console.Read();
            Console.Read();
*/
