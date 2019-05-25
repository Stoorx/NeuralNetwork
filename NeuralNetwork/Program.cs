using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace NeuralNetwork
{
    internal class Program
    {
        private static double TargetFunction(double x) => Math.Sqrt(1 + 4 * x + 12 * x * x);

        public static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(1, 1, 3, 10);
            Console.WriteLine(nn);

            Dataset ds = new Dataset(500,
                x => new List<double> {TargetFunction(x[0])},
                new List<Tuple<double, double>> {new Tuple<double, double>(-10.0, 10.0)}
            );

            File.WriteAllText("dataset.csv", ds.ToString());

            int epochCount = 5000;
            double trainProportion = 0.7;
            int logEveryNEpochs = 1000;
            nn.LearningRate = 0.0005;

            for (int i = 0; i < epochCount; i++)
            {
                Console.WriteLine($"Epoch: {i}/{epochCount.ToString()}");
                var batch = ds.GetBatch(trainProportion);
                foreach (var datapoint in batch)
                {
                    nn.Train(datapoint);
                }

                if (i % logEveryNEpochs == 0 || i == epochCount - 1)
                {
                    File.WriteAllText($"out{i}.csv", ToCsvGraph(nn, ds));
                    Console.WriteLine($"File 'out{i}.csv was written.");
                }
            }
        }

        private static String ToCsvGraph(NeuralNetwork nn, Dataset ds)
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("input, predict, real\n");
            foreach (var datapoint in ds.Datapoints)
            {
                sb.Append($"{datapoint.Item1[0].ToString(CultureInfo.InvariantCulture.NumberFormat)}, ")
                    .Append($"{nn.Calculate(datapoint.Item1)[0].ToString(CultureInfo.InvariantCulture.NumberFormat)}, ")
                    .Append(datapoint.Item2[0].ToString(CultureInfo.InvariantCulture.NumberFormat)).Append('\n');
            }

            return sb.ToString();
        }
    }
}