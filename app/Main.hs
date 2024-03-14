{-# OPTIONS_GHC -Wno-orphans #-}
module Main where
import Data.Functor
import Data.List.Split
import Numeric.LinearAlgebra

-- Check this one for benchmarks: https://www.kaggle.com/code/jedrzejdudzicz/mnist-dataset-100-accuracy

data AFunction = Relu | Sigmoid | SoftMax | Tanh deriving (Show)

data TrainingData = TrainingData { inputs :: Vector Double, label :: Double } deriving (Show)

data Layer = Layer {
  weights :: Vector Double,
  biases :: Vector Double,
  direction :: Vector Double,
  directionStength :: Vector Double,
  activation :: AFunction } deriving (Show)

data NeuralNetwork = NeuralNetwork { indata :: Vector Double, layers :: [Layer] } deriving (Show)

trainingDataFromFile :: FilePath -> IO [TrainingData]
trainingDataFromFile file = do
    nums <- iterateFile file <&> mapToNumbers
    let floats = mapToFloats nums
    let lables::[Double] = map (fromIntegral . head) nums
    return $ zipWith TrainingData floats lables
    where
      mapToFloats :: [[Int]] -> [Vector Double]
      mapToFloats = map (fromList . map ((/255) . fromIntegral) . tail)
      mapToNumbers :: [String] -> [[Int]]
      mapToNumbers = map (map read . splitOn ",")
      iterateFile :: FilePath -> IO [String]
      iterateFile f = do
        content <- readFile f
        return $ tail (lines content)

sigmoid :: Vector Double -> Vector Double
sigmoid = cmap (\x -> 1 / (1 + exp (-x)))

relu :: Vector Double -> Vector Double
relu = cmap (`max` 0)

softMax :: Vector Double -> Vector Double
softMax x = cmap (/ sumElements (cmap exp x)) (cmap exp x)

tanH :: Vector Double -> Vector Double
tanH = cmap tanh

createLayerWithRandomWeights :: Int -> AFunction -> IO Layer
createLayerWithRandomWeights len aFunction = do
    weights <- rand 1 len
    biases <- rand 1 len
    direction <- rand 1 len
    directionStength <- rand 1 len
    return $ Layer (flatten weights) (flatten biases) (directionOneOrMinusOne $ flatten direction) (flatten directionStength) aFunction
    where directionOneOrMinusOne = cmap (\x -> if x > 0.5 then 1 else -1)

main :: IO ()
main = do
    trainingData <- trainingDataFromFile "/var/tmp/mnist/mnist_train.csv"
    print $ length trainingData
