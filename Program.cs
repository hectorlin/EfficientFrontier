using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;

public class StockData
{
    public DateTime Date { get; set; }
    public double Close { get; set; }
}

public class EfficientFrontierPoint
{
    public double Return { get; set; }
    public double Volatility { get; set; }
    public double SharpeRatio { get; set; }
    public List<double> Weights { get; set; } = new();
}

public class Program
{
    private static readonly double RiskFreeRate = 0.02; // 2% risk-free rate
    private static readonly int PortfolioCount = 1000; // Number of portfolios to generate

    public static void Main()
    {
        // Read stock data
        var stocksData = new Dictionary<string, List<StockData>>();
        var stockFiles = Directory.GetFiles(".", "*_daily.csv");
        
        foreach (var file in stockFiles)
        {
            var stockName = Path.GetFileNameWithoutExtension(file).Split('_')[0];
            stocksData[stockName] = ReadStockData(file);
        }

        // Calculate returns for each stock
        var returns = CalculateReturns(stocksData);
        
        // Calculate mean returns and covariance matrix
        var meanReturns = CalculateMeanReturns(returns);
        var covMatrix = CalculateCovarianceMatrix(returns);

        // Generate efficient frontier points
        var efficientFrontier = GenerateEfficientFrontier(meanReturns, covMatrix);

        // Write results to CSV
        WriteEfficientFrontierToCSV(efficientFrontier, stocksData.Keys.ToList());
    }

    private static List<StockData> ReadStockData(string filePath)
    {
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
        };

        using var reader = new StreamReader(filePath);
        using var csv = new CsvReader(reader, config);
        return csv.GetRecords<StockData>().OrderBy(x => x.Date).ToList();
    }

    private static Dictionary<string, List<double>> CalculateReturns(Dictionary<string, List<StockData>> stocksData)
    {
        var returns = new Dictionary<string, List<double>>();
        
        foreach (var (stock, data) in stocksData)
        {
            returns[stock] = new List<double>();
            for (int i = 1; i < data.Count; i++)
            {
                var dailyReturn = (data[i].Close - data[i - 1].Close) / data[i - 1].Close;
                returns[stock].Add(dailyReturn);
            }
        }
        
        return returns;
    }

    private static Vector<double> CalculateMeanReturns(Dictionary<string, List<double>> returns)
    {
        var meanReturns = Vector<double>.Build.Dense(returns.Count);
        int i = 0;
        
        foreach (var stockReturns in returns.Values)
        {
            meanReturns[i] = stockReturns.Average() * 252; // Annualized returns
            i++;
        }
        
        return meanReturns;
    }

    private static Matrix<double> CalculateCovarianceMatrix(Dictionary<string, List<double>> returns)
    {
        int n = returns.Count;
        var matrix = Matrix<double>.Build.Dense(n, n);
        var stocksList = returns.Values.ToList();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var cov = CalculateCovariance(stocksList[i], stocksList[j]) * 252; // Annualized covariance
                matrix[i, j] = cov;
            }
        }

        return matrix;
    }

    private static double CalculateCovariance(List<double> x, List<double> y)
    {
        double xMean = x.Average();
        double yMean = y.Average();
        double sum = 0;
        
        for (int i = 0; i < x.Count; i++)
        {
            sum += (x[i] - xMean) * (y[i] - yMean);
        }
        
        return sum / (x.Count - 1);
    }

    private static List<EfficientFrontierPoint> GenerateEfficientFrontier(Vector<double> meanReturns, Matrix<double> covMatrix)
    {
        var numAssets = meanReturns.Count;
        var points = new List<EfficientFrontierPoint>();
        var random = new Random(42);

        for (int i = 0; i < PortfolioCount; i++)
        {
            var weights = GenerateRandomWeights(numAssets, random);
            var weightVector = Vector<double>.Build.DenseOfArray(weights.ToArray());

            var portfolioReturn = weightVector.DotProduct(meanReturns);
            var portfolioVolatility = Math.Sqrt(weightVector.DotProduct(covMatrix.Multiply(weightVector)));
            var sharpeRatio = (portfolioReturn - RiskFreeRate) / portfolioVolatility;

            points.Add(new EfficientFrontierPoint
            {
                Return = portfolioReturn,
                Volatility = portfolioVolatility,
                SharpeRatio = sharpeRatio,
                Weights = weights
            });
        }

        return points.OrderBy(p => p.Volatility).ToList();
    }

    private static List<double> GenerateRandomWeights(int numAssets, Random random)
    {
        var weights = new List<double>();
        double sum = 0;

        for (int i = 0; i < numAssets; i++)
        {
            var weight = random.NextDouble();
            weights.Add(weight);
            sum += weight;
        }

        // Normalize weights to sum to 1
        for (int i = 0; i < weights.Count; i++)
        {
            weights[i] /= sum;
        }

        return weights;
    }

    private static void WriteEfficientFrontierToCSV(List<EfficientFrontierPoint> points, List<string> stockNames)
    {
        using var writer = new StreamWriter("efficient_frontier.csv");
        using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);

        // Write header
        csv.WriteField("Return");
        csv.WriteField("Volatility");
        csv.WriteField("SharpeRatio");
        foreach (var stock in stockNames)
        {
            csv.WriteField($"{stock}_Weight");
        }
        csv.NextRecord();

        // Write data
        foreach (var point in points)
        {
            csv.WriteField(point.Return);
            csv.WriteField(point.Volatility);
            csv.WriteField(point.SharpeRatio);
            foreach (var weight in point.Weights)
            {
                csv.WriteField(weight);
            }
            csv.NextRecord();
        }
    }
}
