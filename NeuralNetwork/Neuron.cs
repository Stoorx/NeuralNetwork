using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron
    {
        private static readonly Random Random = new Random();

        public List<double> InputWeights;

        public Neuron(int inputsCount = 0)
        {
            InputWeights = new List<double>(inputsCount + 1);
            for (var i = 0; i < inputsCount + 1; i++)
            {
                InputWeights.Add(Random.NextDouble());
            }
        }

//        public  List<double> PreviousWeights = new List<double>();
//        public List<double> Corrections    = new List<double>();
//        public List<double> QSum           = new List<double>();
//
//        private double _sum   = 0;
//        private double _func  = 0;
//        private double _error = 0;
//        private double _delta = 0;
        public override string ToString()
        {
            var sb = new StringBuilder();

            InputWeights.ForEach(w => sb.Append(w).Append(" "));
            return sb.ToString();
        }


        public void AddInputs(int count)
        {
            for (var i = 0; i < count; i++)
            {
                InputWeights.Add(Random.NextDouble());
            }
        }

        public double Calculate(IEnumerable<double> input) =>
            NeuralNetwork.ActivationFunction(input.Select((t, i) => InputWeights[i] * t)
                                                 .Sum() + InputWeights.Last());

        public double CalculateLast(IEnumerable<double> input) => input.Select((t, i) => InputWeights[i] * t)
                                                                      .Sum() + InputWeights.Last();
    }
}