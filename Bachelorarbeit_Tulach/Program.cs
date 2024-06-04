using System;
using MathNet.Numerics.LinearAlgebra.Complex;
using MathNet.Numerics.Statistics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.Analysis;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace Bachelorarbeit_Tulach;
class Program
{
    static string TrainDataPath = "AgeDataTrain.csv";
    static string TestDataPath = "AgeDataTest.csv";
    static string modelPath = "AgeModel.zip";
    static double testfraction = 0.2;
    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext(seed: 0);
        var model = Train(mlContext);
        Evaluate(mlContext, model);
        //TestSinglePrediction(mlContext, model);
    }
    static ITransformer Train(MLContext mlContext)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<BloodTest>(TrainDataPath, hasHeader: true, separatorChar: ',');
        var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "PatientAge")
        .Append(mlContext.Transforms.NormalizeMinMax("BunValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("CreatinineValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("SodiumValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("PotassiumValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("ChlorideValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("CalciumValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("AlbuminValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("AlkPhoValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("AstValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("AltValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("BilirubinValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("CholesterolValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("HdlValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("LdlValue"))
        .Append(mlContext.Transforms.NormalizeMinMax("TriglyceridesValue"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PatientGenderEncoded", inputColumnName: "PatientGender"))
        .Append(mlContext.Transforms.Concatenate("Features", "BunValue", "CreatinineValue", "SodiumValue", "PotassiumValue", "ChlorideValue", "CalciumValue", "AlbuminValue", "AlkPhoValue", "AstValue", "AltValue", "BilirubinValue", "CholesterolValue", "HdlValue", "LdlValue", "TriglyceridesValue", "PatientGenderEncoded"))
        .Append(mlContext.Transforms.DropColumns("PatientId", "SpecimenId", "SampleTime"));
        //.Append(mlContext.Regression.Trainers.OnlineGradientDescent());
        //Append(mlContext.Regression.Trainers.FastForest());
        //.Append(mlContext.Regression.Trainers.Sdca());
        //.Append(mlContext.Regression.Trainers.FastTree());
        //.Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());
        //.Append(mlContext.Regression.Trainers.FastTreeTweedie());
        //.Append(mlContext.Regression.Trainers.Gam());

        //choose trainer by uncommenting
        var preview = dataView.Preview();
        var rowview = dataView.Preview().RowView;
        Console.WriteLine("Training...");
        var model = pipeline.Fit(dataView);
        return model;
    }
    static void Evaluate(MLContext mlContext, ITransformer model)
    {
        Console.WriteLine("calculating evaluation...");
        IDataView data = mlContext.Data.LoadFromTextFile<BloodTest>(TestDataPath, hasHeader: true, separatorChar: ',');
        var predictions = model.Transform(data);
        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "PatientAge", scoreColumnName: "Score");
        Console.WriteLine();
        Console.WriteLine("r-squared: " + metrics.RSquared);
        Console.WriteLine("absolute-loss: " + metrics.MeanAbsoluteError);
        Console.WriteLine("squared-loss: " + metrics.MeanSquaredError);
        Console.WriteLine("RMS-loss: " + metrics.RootMeanSquaredError);

        Console.WriteLine("calculating scatter...");
        float[] trueValues = mlContext.Data.CreateEnumerable<BloodTest>(data, reuseRowObject: false)
                                     .Select(data => data.PatientAge)
        .ToArray();
        float[] predictedValues = mlContext.Data.CreateEnumerable<BloodTestPrediciton>(predictions, reuseRowObject: false)
                                             .Select(prediction => prediction.PatientAge)
                                             .ToArray();
        using (StreamWriter file = new StreamWriter("trueandpredicted.csv"))
        {
            file.WriteLine("TrueValues,PredictedValues"); // Write the header

            for (int i = 0; i < trueValues.Length; i++)
            {
                file.WriteLine($"{trueValues[i]},{predictedValues[i]}");
            }
        }

        Console.WriteLine("calculating residual plot...");
        IEnumerable<(float TrueValue, float PredictedValue)> trueAndPredicted = trueValues.Zip(predictedValues, (t, p) => (t, p));
        List<float> residuals = new List<float>();
        List<float> predictedValuesList = new List<float>();
        foreach (var item in trueAndPredicted)
        {
            float residual = item.TrueValue - item.PredictedValue;
            residuals.Add(residual);
            predictedValuesList.Add(item.PredictedValue);
        }
        using (var writer = new StreamWriter("residuals.csv"))
        {
            writer.WriteLine("PredictedValue,Residual");
            for (int i = 0; i < residuals.Count; i++)
            {
                writer.WriteLine($"{predictedValues[i]},{residuals[i]}");
            }
        }

        Console.WriteLine("calculating PFI...");
        var permutationFeatureImportance = mlContext.Regression.PermutationFeatureImportance(model, predictions, permutationCount: 5, labelColumnName: "PatientAge");
        foreach (var item in permutationFeatureImportance)
        {
            Console.WriteLine("key: " + item.Key);
            Console.WriteLine("mean lossfunc: " + item.Value.LossFunction.Mean);
            Console.WriteLine("mean meanabserror: " + item.Value.MeanAbsoluteError.Mean);
            Console.WriteLine("mean meansqderror: " + item.Value.MeanSquaredError.Mean);
            Console.WriteLine("mean rms: " + item.Value.RootMeanSquaredError.Mean);
            Console.WriteLine("mean r2: " + item.Value.RSquared.Mean);
            Console.WriteLine();
            Console.WriteLine("stdev lossfunc: " + item.Value.LossFunction.StandardDeviation);
            Console.WriteLine("stdev meanabserror: " + item.Value.MeanAbsoluteError.StandardDeviation);
            Console.WriteLine("stdev meansqderror: " + item.Value.MeanSquaredError.StandardDeviation);
            Console.WriteLine("stdev rms: " + item.Value.RootMeanSquaredError.StandardDeviation);
            Console.WriteLine("stdev r2: " + item.Value.RSquared.StandardDeviation);
            Console.WriteLine();
            Console.WriteLine("stderr lossfunc: " + item.Value.LossFunction.StandardError);
            Console.WriteLine("stderr meanabserror: " + item.Value.MeanAbsoluteError.StandardError);
            Console.WriteLine("stderr meansqderror: " + item.Value.MeanSquaredError.StandardError);
            Console.WriteLine("stderr rms: " + item.Value.RootMeanSquaredError.StandardError);
            Console.WriteLine("stderr r2: " + item.Value.RSquared.StandardError);
            Console.WriteLine();
            Console.WriteLine("------------------");
        }
    }
}