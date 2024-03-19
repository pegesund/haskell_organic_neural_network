{-# OPTIONS_GHC -Wno-orphans #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use newtype instead of data" #-}
{-# HLINT ignore "Use join" #-}
module Main where
import Data.Functor
import Data.List.Split
import Numeric.LinearAlgebra
import Data.List
import System.Random
import Numeric.LinearAlgebra.Devel (zipVectorWith)
import Control.Monad

-- import UnliftIO
-- import Control.Concurrent

-- Check this one for benchmarks: https://www.kaggle.com/code/jedrzejdudzicz/mnist-dataset-100-accuracy

data AFunction = Relu | Sigmoid | SoftMax | Tanh deriving (Show)

data TrainingData = TrainingData { inputs :: Vector Double, label :: Double } deriving (Show)

data TrainingParameters = TraininParameters {
  epochs :: Int,
  batchSize :: Int,
  maxChildren :: Int,
  maxEphochLifes :: Int,
  devSpeed :: Double,
  lossFunction :: Vector Double -> Vector Double -> Double,
  accuracyFunction :: Vector Double -> Vector Double -> Double,
  changeDir :: Int,
  numberOfChildren :: Int
  }

instance Show TrainingParameters where
  show (TraininParameters epochs' batchSize' maxChildren' maxEphochLifes' devSpeed' _ _ _ _) = "TraininParameters\n" ++ "epochs: " ++ show epochs' ++ "\n" ++ "batchSize: " ++ show batchSize' ++ "\n" ++ "maxChildren: " ++ show maxChildren' ++ "\n" ++ "maxEphochLifes: " ++ show maxEphochLifes' ++ "\n" ++ "devSpeed: " ++ show devSpeed'

data Layer = Layer {
  weights :: Vector Double,
  direction :: Vector Double,
  directionStength :: Vector Double,
  activation :: AFunction } deriving (Show)

data NeuralNetwork = NeuralNetwork { biases :: Vector Double, layers :: [Layer] }

data TrainedNetwork = TrainedNetwork {
  nn' :: NeuralNetwork,
  generation :: Int,
  accuracy :: Double } deriving (Show)


-- implement show for NeuralNetwork, each layer should be printed with its weights and biases and activation function on a new line
instance Show NeuralNetwork where
  show (NeuralNetwork biases' layers') = "NeuralNetwork\n" ++ "biases: " ++ show biases' ++ "\n" ++ "layers: " ++ show layers'

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
    direction <- rand 1 len
    directionStength <- rand 1 len
    return $ Layer (flatten weights) (directionOneOrMinusOne $ flatten direction) (flatten directionStength) aFunction
    where directionOneOrMinusOne = cmap (\x -> if x > 0.5 then 1 else -1)

createNeuralNetwork :: [(Int, AFunction)] -> IO NeuralNetwork
createNeuralNetwork layers = do
    -- biases' <- do flatten <$> rand 1 (fst $ head layers)
    let biases' = konst 0 (fst $ head layers)
    layers' <- mapM (uncurry createLayerWithRandomWeights) layers
    return $ NeuralNetwork biases' layers'

feedForward :: NeuralNetwork -> Vector Double -> Vector Double
feedForward (NeuralNetwork _ layers) indata = foldl' feedForwardLayer indata layers
    where
      feedForwardLayer :: Vector Double -> Layer -> Vector Double
      feedForwardLayer indata' (Layer weights'  _ _ activation') = case activation' of
        Relu -> sumWeights relu
        Sigmoid -> sumWeights sigmoid
        SoftMax -> sumWeights softMax
        Tanh -> sumWeights tanH
        where sumWeights func = func $ let allWeightsAndBiases theWeight = sumElements $ cmap (+ theWeight) indata'
                                          in cmap allWeightsAndBiases weights'

main :: IO ()
main = do
    trainingData <- trainingDataFromFile "/var/tmp/mnist/mnist_train.csv"
    print $ length trainingData
    testme

testme :: IO ()
testme = do
    let indata = fromList [1, 2, 3]
    let layers = [(2, Relu), (1, Relu)]
    nn <- createNeuralNetwork layers
    print nn
    print $ feedForward nn indata
    

updateWeights :: NeuralNetwork -> TrainingParameters -> IO NeuralNetwork
updateWeights nn traininParameters = do
  let layers' = layers nn
  newLayers <- mapM (updateLayerWeights traininParameters) layers'
  return $ NeuralNetwork (biases nn) newLayers
  where
    updateLayerWeights :: TrainingParameters -> Layer -> IO Layer
    updateLayerWeights trainingParameters' (Layer weights' direction' directionStength' activation') = do
      direction'' <- changeDirection
      directionStrength'' <- changeDirectionStrength
      let weights'' = weights' * directionStrength''
      return $ Layer weights'' direction''  directionStrength'' activation'
      where
        changeDirection :: IO (Vector Double)
        changeDirection = do
          rDir <- randomRIO (0, changeDir trainingParameters') :: IO Int
          return $ if rDir == 1 then direction' * (-1) else direction'
        changeDirectionStrength :: IO (Vector Double)
        changeDirectionStrength = do
          rDirStrength <- randomRIO (0, 0.1) :: IO Double
          return $ cmap (\x -> x * ( 1 + rDirStrength)) directionStength' * direction'

mergeNetworksAddition :: NeuralNetwork -> NeuralNetwork -> NeuralNetwork
mergeNetworksAddition nn1 nn2 = NeuralNetwork (biases nn1) (zipWith mergeLayers (layers nn1) (layers nn2))
  where
    mergeLayers :: Layer -> Layer -> Layer
    mergeLayers (Layer weights1 direction1 directionStength1 activation) (Layer weights2 direction2 directionStength2 _) =
      Layer
      ((weights1 + weights2) / 2)
      ((direction1 + direction2) /2)
      ((directionStength1 + directionStength2) / 2)
      activation


mergeNetworkRandom :: NeuralNetwork -> NeuralNetwork -> IO NeuralNetwork
mergeNetworkRandom nn1 nn2 = do
  let layers1 = layers nn1
  let layers2 = layers nn2
  newLayers <- zipWithM mergeLayers layers1 layers2
  return $ NeuralNetwork (biases nn1) newLayers
  where
    mergeLayers :: Layer -> Layer -> IO Layer
    mergeLayers (Layer weights1 direction1 directionStength1 activation) (Layer weights2 direction2 directionStength2 _) = do
      r <- random01Vector (size weights1)
      let r' = inverse01Vector r
      let weights' = weights1 * r + weights2 * r'
      let direction' = direction1 * r + direction2 * r'
      let directionStength' = directionStength1 * r + directionStength2 * r'
      return $ Layer weights' direction' directionStength' activation where
        random01Vector :: Int -> IO (Vector Double)
        random01Vector n = do 
          r <- rand 1 n
          return $ roundVector ( flatten r) 
        inverse01Vector :: Vector Double -> Vector Double
        inverse01Vector = cmap (\x -> if x == 0 then 1 else 0)
        






