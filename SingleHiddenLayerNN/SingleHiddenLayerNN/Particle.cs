using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SingleHiddenLayerNN
{
    class Particle
    {
        public double[] position;
        public double error;

        public double[] velocity;

        public double[] bestPosition;
        public double bestError;

        public Particle(double[] position, double error, double[] velocity, double[] bestPosition, double bestError)
        {
            this.position = new double[position.Length];
            position.CopyTo(this.position, 0);

            this.error = error;

            this.velocity = new double[velocity.Length];
            velocity.CopyTo(this.velocity, 0);

            this.bestPosition = new double[bestPosition.Length];
            bestPosition.CopyTo(this.bestPosition, 0);

            this.bestError = bestError;
        }
    }
}
