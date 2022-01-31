using System;

namespace SingleHiddenLayerNN
{
    class Program
    {
        static void Main(string[] args)
        {
            Data data = new Data();
            data.Info();

            ShowData showData = new ShowData();

            Console.WriteLine("\n\t\tAll raw data\n");
            showData.Show(data.rawData, 5, 2, true);

            showData.SplitData(data.rawData, 37, out double[][] train, out double[][] test);

            Console.WriteLine("Train data:");
            showData.Show(train, 5, 2, true);
            Console.WriteLine("\nTest data:");
            showData.Show(test, 2, 2, true);
        }
    }
}
