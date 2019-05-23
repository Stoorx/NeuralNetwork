using System;

namespace NeuralNetwork
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(1, 1, 3, 10);
            Console.WriteLine(nn);
        }
    }
}