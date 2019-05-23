using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private int _inputsCount;
        private NeuronLayer _lastLayer;

        private List<NeuronLayer> _layers = new List<NeuronLayer>();

        public NeuralNetwork(int inputsCount, int outputCount, int layersCount, int layerSize)
        {
            _lastLayer = new NeuronLayer(outputCount);
            _lastLayer.AddInputs(layerSize);

            _inputsCount = inputsCount;

            var firstLayer = new NeuronLayer(layerSize);
            firstLayer.AddInputs(inputsCount);
            _layers.Add(firstLayer);

            for (var i = 1; i < layersCount; i++)
            {
                var nl = new NeuronLayer(layerSize);
                nl.AddInputs(layerSize);
                _layers.Add(nl);
            }
        }

        public double LearningRate { get; set; }
        public double Moment { get; set; }

        public static double ActivationFunction(double x) =>
            1.0 / (1.0 + Math.Exp(-x));

        public static double ActivationFunctionDerivative(double x) =>
            ActivationFunction(x) * (1 - ActivationFunction(x));

        public List<double> Calculate(List<double> input)
        {
            if (input.Count != _inputsCount)
                throw new Exception("Inputs count does not match NN inputs.");
            var currInput = input;
            foreach (var layer in _layers)
            {
                currInput = layer.Calculate(currInput);
            }

            return _lastLayer.Calculate(currInput, true);
        }


        public override string ToString()
        {
            var sb = new StringBuilder();

            for (int i = 0; i < _layers.Count; i++)
            {
                sb.Append($"Layer {i}:\n").Append(_layers[i]);
            }

            sb.Append("Output layer:\n").Append(_lastLayer);
            return sb.ToString();
        }
    }
}