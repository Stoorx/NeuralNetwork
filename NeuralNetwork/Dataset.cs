using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Dataset
    {
        private static readonly Random Rnd = new Random();
        public readonly List<Tuple<List<double>, List<double>>> Datapoints;

        public Dataset(List<Tuple<List<double>, List<double>>> datapoints)
        {
            Datapoints = datapoints;
        }

        public Dataset(int size, Func<List<double>, List<double>> func, List<Tuple<double, double>> bounds)
        {
            Datapoints = new List<Tuple<List<double>, List<double>>>();

            for (int i = 0; i < size; i++)
            {
                List<double> inputs = new List<double>(bounds.Count);
                for (int j = 0; j < bounds.Count; j++)
                {
                    inputs.Add((bounds[j].Item2 - bounds[j].Item1) * Rnd.NextDouble() + bounds[j].Item1);
                }

                Datapoints.Add(new Tuple<List<double>, List<double>>(inputs, func(inputs)));
            }
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach (var datapoint in Datapoints)
            {
                foreach (var input in datapoint.Item1)
                {
                    sb.Append(input).Append(' ');
                }

                sb.Append("; ");

                foreach (var output in datapoint.Item2)
                {
                    sb.Append(output).Append(' ');
                }

                sb.Append('\n');
            }

            return sb.ToString();
        }

        public List<Tuple<List<double>, List<double>>> GetBatch(double proportion)
        {
            var batch = new List<Tuple<List<double>, List<double>>>();
            foreach (var datapoint in Datapoints)
            {
                if (Rnd.NextDouble() < proportion)
                {
                    batch.Add(datapoint);
                }
            }

            return batch;
        }
    }
}