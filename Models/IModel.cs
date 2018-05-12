using MathNet.Numerics.LinearAlgebra;

namespace SharpNeuralNetworks.Models
{
    public interface IModel
    {

        void Fit(Matrix<double> x, Vector<double> y, bool addOnes = true, bool verbose = false);

        double Predict(Vector<double> x);

        Vector<double> Predict(Matrix<double> x, bool addOnes = true);

        double Score(Matrix<double> x, Vector<double> y);

    }
}
