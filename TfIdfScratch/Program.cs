using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TfIdfScratch
{
    class Program
    {
        static void Main(string[] args)
        {

            var texts = new List<Text> {
                new Text { Data = "apple apple orange grape" },
                new Text { Data = "grape apple melon" },
                 new Text { Data = "grape banana melon" }
            };

            var ml = new MLContext();
            var data = ml.Data.LoadFromEnumerable(texts);
            var textFeaturizingOptions = new TextFeaturizingEstimator.Options
            {
                KeepDiacritics = false,
                KeepPunctuations = false,
                KeepNumbers = false,
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options(),
                WordFeatureExtractor = new WordBagEstimator.Options()
                {
                    Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf
                },
                CharFeatureExtractor = null
            };
            var vectorizer = ml.Transforms.Text.FeaturizeText("TfIDFWeights", options: textFeaturizingOptions, inputColumnNames: "Data");
            var result = vectorizer.Fit(data).Transform(data);
            var column = result.GetColumn<VBuffer<float>>("TfIDFWeights");
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            result.Schema["TfIDFWeights"].GetSlotNames(slotNames: ref slotNames);
            var words = slotNames.DenseValues().ToArray();
            var doc = 0;
            foreach (var tfidf in column)
            {
                for (int i = 0; i < tfidf.Length; i++)
                    Console.WriteLine($"doc:{doc} word '{words[i]}' {tfidf.GetItemOrDefault(i)}");
                doc++;
            }
            Console.ReadLine();
        }
    }
    class Text
    {
        public string Data;
    }
}
