using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SingleHiddenLayerNN
{
    class NeuralNetwork
    {
        private int numInput;         //number of nodes for input layer
        private int numHidden;        //number of nodes for hidden layer
        private int numOutput;        //number of nodes for output layer

        private double[] inputs;      //input values
        private double[][] ihWeights; //weights from input to hidden
        private double[] hBiases;     //biases of hidden
        private double[] hOutputs;    //outputs of hidden

        private double[][] hoWeights; //weights from hidden to output
        private double[] oBiases;     //biases of output
        private double[] outputs;     //outputs of output

        private Random rand;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            inputs = new double[numInput];

            ihWeights = MakeMatrix(numInput, numHidden);
            hBiases = new double[numHidden];
            hOutputs = new double[numHidden];

            hoWeights = MakeMatrix(numHidden, numOutput);
            oBiases = new double[numOutput];
            outputs = new double[numOutput];

            rand = new Random(0);
        }

        private double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < result.Length; i++)
                result[i] = new double[cols];
            return result;
        }

        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput + 1) * numHidden +
                             (numHidden + 1) * numOutput;

            if (weights.Length != numWeights)
                throw new Exception("Incorrect weights array length");

            int k = 0;

            //picks all weights and biases accordingly for hidden and output layers

            for (int i = 0; i < numInput; i++)
                for (int j = 0; j < numHidden; j++)
                    ihWeights[i][j] = weights[k++];

            for (int i = 0; i < numHidden; i++)
                hBiases[i] = weights[k++];

            for (int i = 0; i < numHidden; i++)
                for (int j = 0; j < numOutput; j++)
                    hoWeights[i][j] = weights[k++];

            for (int i = 0; i < numOutput; i++)
                oBiases[i] = weights[k++];
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden];
            double[] oSums = new double[numOutput];

            for (int i = 0; i < xValues.Length; i++)
                inputs[i] = xValues[i];

            for (int i = 0; i < numHidden; i++)
                for (int j = 0; j < numInput; j++)
                    hSums[i] += ihWeights[j][i] * inputs[j];

            for (int i = 0; i < numHidden; i++)
                hSums[i] += hBiases[i];

            for (int i = 0; i < numHidden; i++)
                hOutputs[i] = HyperTan(hSums[i]);

            for (int i = 0; i < numOutput; i++)
                for (int j = 0; j < numHidden; j++)
                    oSums[i] += hoWeights[j][i] * hOutputs[j];

            for (int i = 0; i < numOutput; i++)
                oSums[i] += oBiases[i];

            double[] softmaxOutputs = Softmax(oSums);
            Array.Copy(softmaxOutputs, outputs, softmaxOutputs.Length);

            double[] result = new double[numOutput];
            Array.Copy(outputs, result, result.Length);

            return result;
        }

        private double HyperTan(double v)
        {
            if (v < -20.0)
                return -1;
            else if (v > 20.0)
                return 1;
            else
                return Math.Tanh(v);
        }

        private double[] Softmax(double[] oSums)
        {
            double max = oSums[0];

            for (int i = 0; i < oSums.Length; i++)
                if(max < oSums[i]) max = oSums[i];

            double scale = 0.0;

            for (int i = 0; i < oSums.Length; i++)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];

            for (int i = 0; i < oSums.Length; i++)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result;
        }

        public double[] Train(double[][] trainData, int nParticles, int nEpochs)
        {
            int numWeights = (numInput + 1) * numHidden + (numHidden + 1) * numOutput;

            int epoch = 0;

            double minX = -10.0;
            double maxX = 10.0;
            double w = 0.729;
            double c1 = 1.49445;
            double c2 = 1.49445;
            double r1, r2;

            Particle[] swarm = new Particle[nParticles];

            double[] bestGlobalPosition = new double[numWeights];

            double bestGlobalError = double.MaxValue;

            double lo = 0.1 * minX;
            double hi = 0.1 * maxX;

            for (int i = 0; i < swarm.Length; i++)
            {
                double[] randomPosition = new double[numWeights];

                for (int j = 0; j < randomPosition.Length; j++)
                    randomPosition[j] = (maxX - minX) * rand.NextDouble() + minX;

                double error = MeanSquareError(trainData, randomPosition);

                double[] randomVelocity = new double[numWeights];

                for (int k = 0; k < randomVelocity.Length; k++)
                    randomVelocity[k] = (hi - lo) * rand.NextDouble() + lo;

                swarm[i] = new Particle(randomPosition, error, randomVelocity, randomPosition, error);

                if (swarm[i].error < bestGlobalError)
                {
                    bestGlobalError = swarm[i].error;
                    swarm[i].position.CopyTo(bestGlobalPosition, 0);
                }
            }
            //--------Partical swarm optimization---------
            int[] sequence = new int[nParticles];

            for (int i = 0; i < sequence.Length; i++)
                sequence[i] = i;

            while (epoch < nEpochs)
            {
                double[] newVelocity = new double[numWeights];
                double[] newPosition = new double[numWeights];
                double newError;

                Shuffle(sequence);

                for (int j = 0; j < swarm.Length; j++)
                {
                    int index = sequence[j];

                    Particle currentParticle = swarm[index];

                    //NEW VELOCITY
                    for (int k = 0; k < currentParticle.velocity.Length; k++)
                    {
                        r1 = rand.NextDouble();
                        r2 = rand.NextDouble();

                        newVelocity[k] = (currentParticle.velocity[k] * w) +
                                         ((currentParticle.bestPosition[k] - currentParticle.position[k]) * r1 * c1) +
                                         ((bestGlobalPosition[k] - currentParticle.position[k]) * r2 * c2);
                    }
                    newVelocity.CopyTo(currentParticle.velocity, 0);

                    //NEW POSITION
                    for (int p = 0; p < currentParticle.position.Length; p++)
                    {
                        newPosition[p] = currentParticle.position[p] + newVelocity[p];

                        if (newPosition[p] < minX)
                            newPosition[p] = minX;
                        else if (newPosition[p] > maxX)
                            newPosition[p] = maxX;
                    }
                    newPosition.CopyTo(currentParticle.position, 0);

                    //ERROR
                    newError = MeanSquareError(trainData, newPosition);

                    currentParticle.error = newError;

                    if (newError < currentParticle.bestError)
                    {
                        currentParticle.bestError = newError;
                        newPosition.CopyTo(currentParticle.bestPosition, 0);
                    }
                    if (newError < bestGlobalError)
                    {
                        bestGlobalError = newError;
                        newPosition.CopyTo(bestGlobalPosition, 0);
                    }
                }
                epoch++;
            }
            SetWeights(bestGlobalPosition);

            double[] result = new double[numWeights];
            Array.Copy(bestGlobalPosition, result, result.Length);

            return result;
        }

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; i++)
            {
                int index = rand.Next(i, sequence.Length);
                int temp = sequence[index];
                sequence[index] = sequence[i];
                sequence[i] = temp;
            }
        }

        private double MeanSquareError(double[][] trainData, double[] weights)
        {
            SetWeights(weights);

            double[] xVal = new double[numInput];
            double[] tVal = new double[numOutput];

            double sumSquaredError = 0.0;

            for (int i = 0; i < trainData.Length; i++)
            {
                Array.Copy(trainData[i], xVal, numInput);               //first 4
                Array.Copy(trainData[i], numInput, tVal, 0, numOutput); //last 3

                double[] yVal = ComputeOutputs(xVal);
                for (int j = 0; j < yVal.Length; j++)
                    sumSquaredError += ((yVal[j] - tVal[j]) * (yVal[j] - tVal[j]));
            }
            return sumSquaredError / trainData.Length;
        }

        public double Accuracy(double[][] testData)
        {
            int correct = 0;
            int wrong = 0;

            double[] xVal = new double[numInput];
            double[] tVal = new double[numInput];
            double[] yVal;

            for (int i = 0; i < testData.Length; i++)
            {
                Array.Copy(testData[i], xVal, numInput);
                Array.Copy(testData[i], numInput, tVal, 0, numOutput);

                yVal = ComputeOutputs(xVal);
                int indexOfMax = MaxIndex(yVal);

                if (tVal[indexOfMax] == 1.0)
                    correct++;
                else
                    wrong++;
            }
            return (correct * 1.0) / (correct + wrong);
        }

        private int MaxIndex(double[] yVal)
        {
            int indexOfMax = 0;
            double biggestVal = yVal[0];

            for (int i = 0; i < yVal.Length; i++)
            {
                if (yVal[i] > biggestVal)
                {
                    biggestVal = yVal[i];
                    indexOfMax = i;
                }
            }
            return indexOfMax;
        }
    }
}
