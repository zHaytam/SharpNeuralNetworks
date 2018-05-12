using System;
using MathNet.Numerics.LinearAlgebra;
using SharpNeuralNetworks.Models;

namespace SharpNeuralNetworks.GradientDescent
{
    public class StochasticGradientDescent : IGradientDescent
    {

        #region Properties

        public IModel Model { get; set; }

        public double LearningRate { get; set; }

        public int MaxIters { get; set; }

        public Vector<double> Weights { get; set; }

        #endregion

        public StochasticGradientDescent(IModel model, Vector<double> initialWeights, double learningRate = 0.01, int maxIters = 100)
        {
            Model = model;
            Weights = initialWeights;
            LearningRate = learningRate;
            MaxIters = maxIters;
        }

        #region Public Methods

        public void Train(Vector<double> x, double y)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x));

            if (x.Count != Weights.Count)
                throw new ArgumentException($"The weights and {nameof(x)} must have the same count ({Weights.Count} vs {x.Count}).");

            double error = y - Model.Predict(x);

            for (int j = 0; j < Weights.Count; j++)
            {
                Weights[j] = Weights[j] + error * LearningRate * x[j];
            }
        }

        public void Train(Matrix<double> x, Vector<double> y, bool addOnes = false, bool earlyStop = true, bool verbose = false)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x));

            if (y == null)
                throw new ArgumentNullException(nameof(y));

            if (x.RowCount != y.Count)
                throw new ArgumentException($"The input matrix's row count and the target's count must be the same ({x.ToTypeString()} vs {y.Count}).");

            if (addOnes)
            {
                x = x.InsertColumn(0, Vector<double>.Build.Dense(x.RowCount, 1.0));
            }

            for (int i = 0; i < MaxIters; i++)
            {
                double score = Model.Score(x, y);
                if (score == 1.0)
                    break;

                if (verbose) Console.WriteLine($"Iter #{i + 1}, Score: {score}");

                for (int j = 0; j < x.RowCount; j++)
                {
                    Train(x.Row(j), y[j]);
                }
            }
        }

        #endregion

    }
}
