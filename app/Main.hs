{-# OPTIONS_GHC -Wno-orphans #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use newtype instead of data" #-}
module Main where
import Data.Functor
import Data.List.Split
import Numeric.LinearAlgebra
import Data.List

-- Check this one for benchmarks: https://www.kaggle.com/code/jedrzejdudzicz/mnist-dataset-100-accuracy

data AFunction = Relu | Sigmoid | SoftMax | Tanh deriving (Show)

data TrainingData = TrainingData { inputs :: Vector Double, label :: Double } deriving (Show)

data Layer = Layer {
  weights :: Vector Double,
  biases :: Vector Double,
  direction :: Vector Double,
  directionStength :: Vector Double,
  activation :: AFunction } deriving (Show)

data NeuralNetwork = NeuralNetwork { indata :: Vector Double, layers :: [Layer] } 

-- implement show for NeuralNetwork, each layer should be printed with its weights and biases and activation function on a new line
instance Show NeuralNetwork where
  show (NeuralNetwork indata' layers') = "NeuralNetwork\n" ++ "indata: " ++ show indata' ++ "\n" ++ concatMap (\x -> show x ++ "\n") layers'

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


createNeuralNetwork :: Vector Double -> [(Int, AFunction)] -> IO NeuralNetwork
createNeuralNetwork indata layers = do
    layers' <- mapM (uncurry createLayerWithRandomWeights) layers
    return $ NeuralNetwork indata layers'


feedForward :: NeuralNetwork -> Vector Double
feedForward (NeuralNetwork indata layers) = foldl' feedForwardLayer indata layers
    where
      feedForwardLayer :: Vector Double -> Layer -> Vector Double
      feedForwardLayer indata' (Layer weights' biases' _ _ activation') = case activation' of
        Relu -> sumWeights relu
        Sigmoid -> sumWeights sigmoid
        SoftMax -> sumWeights softMax
        Tanh -> sumWeights tanH
        where sumWeights func = func $ let allWeightsAndBiases theWeight = sumElements $ cmap (+ theWeight) indata'
                                          in cmap allWeightsAndBiases weights' + biases'

main :: IO ()
main = do
    trainingData <- trainingDataFromFile "/var/tmp/mnist/mnist_train.csv"
    print $ length trainingData

testme :: IO ()
testme = do
    let indata = fromList [1, 2, 3]
    let layers = [(2, Relu), (1, Relu)]
    nn <- createNeuralNetwork indata layers
    print nn
    print $ feedForward nn
