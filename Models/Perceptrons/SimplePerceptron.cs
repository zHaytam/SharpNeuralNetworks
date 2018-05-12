using System;
using MathNet.Numerics.LinearAlgebra;
using SharpNeuralNetworks.GradientDescent;

namespace SharpNeuralNetworks.Models.Perceptrons
{
    public class SimplePerceptron : IModel
    {

        #region Fields

        private readonly StochasticGradientDescent _sgd;

        #endregion

        #region Properties

        public Vector<double> Weights => _sgd.Weights;

        #endregion

        public SimplePerceptron(Vector<double> initialWeights, double learningRate = 0.01, int maxIters = 100)
        {
            _sgd = new StochasticGradientDescent(this, initialWeights, learningRate, maxIters);
        }

        #region Public Methods

        public void Fit(Matrix<double> x, Vector<double> y, bool addOnes = true, bool verbose = false) => _sgd.Train(x, y, addOnes, true, verbose);

        public double Predict(Vector<double> x)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x));

            double dot = x.DotProduct(Weights);
            return dot > 0 ? 1 : 0;
        }

        public Vector<double> Predict(Matrix<double> x, bool addOnes = true)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x));

            if (addOnes)
            {
                x = x.InsertColumn(0, Vector<double>.Build.Dense(x.RowCount, 1.0));
            }

            var pred = Vector<double>.Build.Dense(x.RowCount);

            for (int i = 0; i < x.RowCount; i++)
            {
                pred[i] = Predict(x.Row(i));
            }

            return pred;
        }

        public double Score(Matrix<double> x, Vector<double> y)
        {
            double rightPreds = 0;
            var preds = Predict(x);

            for (int i = 0; i < preds.Count; i++)
            {
                if (preds[i] == y[i])
                    rightPreds++;
            }

            return rightPreds / preds.Count;
        }

        #endregion

    }
}