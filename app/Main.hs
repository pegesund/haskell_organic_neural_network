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
import Control.Monad
import Data.Function






-- import UnliftIO
-- import Control.Concurrent

-- Check this one for benchmarks: https://www.kaggle.com/code/jedrzejdudzicz/mnist-dataset-100-accuracy

data AFunction = Relu | Sigmoid | SoftMax | Tanh deriving (Show)

data TrainingData = TrainingData { inputs :: Vector Double, label :: Vector Double } deriving (Show)

data TrainingParameters = TraininParameters {
  epochs :: Int,
  batchSize :: Int,
  maxChildren :: Int,
  maxEphochLifes :: Int,
  devSpeed :: Double,
  lossFunction :: Vector Double -> Vector Double -> Double,
  accuracyFunction :: Vector Double -> Vector Double -> Double,
  changeDir :: Int,
  numberOfChildren :: Int,
  extraPowerToEscapeLocalMinima :: Double,
  changesForExtraPower :: Int,
  keepOldBatchesNum :: Int
  }

instance Show TrainingParameters where
  show (TraininParameters epochs' batchSize' maxChildren' maxEphochLifes' devSpeed' _ _ _ _ _ _ _)  = "TraininParameters\n" ++ "epochs: " ++ show epochs' ++ "\n" ++ "batchSize: " ++ show batchSize' ++ "\n" ++ "maxChildren: " ++ show maxChildren' ++ "\n" ++ "maxEphochLifes: " ++ show maxEphochLifes' ++ "\n" ++ "devSpeed: " ++ show devSpeed'

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


data NeuralNetworkWithLoss = NeuralNetworkWithLoss {
  nn :: NeuralNetwork,
  loss :: Double } deriving (Show)


-- implement show for NeuralNetwork, each layer should be printed with its weights and biases and activation function on a new line
instance Show NeuralNetwork where
  show (NeuralNetwork biases' layers') = "NeuralNetwork\n" ++ "biases: " ++ show biases' ++ "\n" ++ "layers: " ++ show layers'


createDefaultTrainingParameters :: TrainingParameters
createDefaultTrainingParameters = TraininParameters {
  epochs = 100,
  batchSize = 300,
  maxChildren = 20,
  maxEphochLifes = 10,
  devSpeed = 1,
  lossFunction = \x y -> sumElements $ cmap (** 2) (abs (x - y)),
  accuracyFunction = \x y -> sumElements $ cmap (** 2) (x - y),
  changeDir = 10,
  numberOfChildren = 10,
  extraPowerToEscapeLocalMinima = 10,
  changesForExtraPower = 5,
  keepOldBatchesNum = 2
  }



trainingDataFromFile :: FilePath -> IO [TrainingData]
trainingDataFromFile file = do
    nums <- iterateFile file <&> mapToNumbers
    let floats = mapToFloats nums
    let lables::[Int] = map (fromIntegral . head) nums
    let labelVectors = map (normalizeDoubleVectorToMax . createLabelVector 10) lables
    return $ zipWith TrainingData floats labelVectors
    where
      mapToFloats :: [[Int]] -> [Vector Double]
      mapToFloats = map (fromList . map ((/255) . fromIntegral) . tail)
      mapToNumbers :: [String] -> [[Int]]
      mapToNumbers = map (map read . splitOn ",")
      iterateFile :: FilePath -> IO [String]
      iterateFile f = do
        content <- readFile f
        return $ tail (lines content)
      createLabelVector :: Int -> Int -> Vector Double
      createLabelVector n l = fromList $ replicate l 0 ++ [1] ++ replicate (n - l - 1) 0

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
        Relu -> normalizeDoubleVectorToMax $ sumWeights relu
        Sigmoid -> normalizeDoubleVectorToMax $ sumWeights sigmoid
        SoftMax -> sumWeights softMax
        Tanh -> normalizeDoubleVectorToMax $ sumWeights tanH
        where sumWeights func = func $ let allWeightsAndBiases theWeight = sumElements $ cmap (* theWeight) indata'
                                          in cmap allWeightsAndBiases weights'

main :: IO ()
main = do
    testTraining

testme :: IO ()
testme = do
    let indata = fromList [1, 2, 3]
    let layers = [(20, Relu), (1, Relu)]
    nn <- createNeuralNetwork layers
    print nn
    print $ feedForward nn indata

normalizeDoubleVectorToMax :: Vector Double -> Vector Double
normalizeDoubleVectorToMax v = cmap (/ m) v where m = maxElement v

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
          tryToEscapeLocalMinima <- randomRIO (0, changesForExtraPower trainingParameters') :: IO Int
          let directionStengthEscape = if tryToEscapeLocalMinima == 1 then  extraPowerToEscapeLocalMinima trainingParameters' else 1
          return $ cmap (\x -> x * ( 1 + (rDirStrength * devSpeed trainingParameters' * directionStengthEscape))) directionStength' * direction'

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


createChildrenFromNetwork :: Int -> TrainingParameters -> NeuralNetwork -> IO [NeuralNetwork]
createChildrenFromNetwork n trainingParameters nn = do
  replicateM n (updateWeights nn trainingParameters)


pickBestSurvivors :: [[TrainingData]] -> [TrainingData] -> [NeuralNetwork] -> [NeuralNetwork] -> TrainingParameters -> IO [NeuralNetwork]
pickBestSurvivors oldBatches newBatch parents children trainingParameters = do
  let concatedBatches = concat $ oldBatches ++ [newBatch]
  concatedNetworks <- sequence $ concatMap (\x -> map (mergeNetworkRandom x) children) parents
  let resultVector :: [[Vector Double]] = map (\child -> map (feedForward child . inputs) concatedBatches) concatedNetworks
  let losses = do
        result <- resultVector
        let sumLosses = sum $ concatMap (\res -> map (lossFunction trainingParameters res . label) concatedBatches) result
        return sumLosses
  neuralNetworksWithLosses <- zipWithM (\x y -> return $ NeuralNetworkWithLoss x y) concatedNetworks losses
  let sorted = sortOn loss neuralNetworksWithLosses
  let best = take (maxChildren trainingParameters) sorted
  return $ map nn best



trainNetworks :: [[TrainingData]] -> [TrainingData] -> [NeuralNetwork] -> TrainingParameters -> IO ([NeuralNetwork], [TrainingData])
trainNetworks oldBatches trainingData' networks trainingParameters' = do
  children <- mapM (createChildrenFromNetwork (numberOfChildren trainingParameters') trainingParameters') networks
  let children' = concat children ++ networks
  -- pick batch size random training data
  randomGenerator <- getStdGen
  let indexes = take (batchSize trainingParameters') $ randomRs (0, length trainingData' - 1) randomGenerator
  -- print indexes
  let batch = map (trainingData' !!) indexes
  -- run feedForward on each of the children with all the batches
  let resultVector :: [[Vector Double]] = map (\child -> map (feedForward child . inputs) batch) children'
  -- calculate the loss for each of the children
  let losses = do
        result <- resultVector
        let inputs = map label batch
        let sumLosses = sum $ concatMap (\res -> map (lossFunction trainingParameters' res) inputs) result
        return sumLosses
  neuralNetworksWithLosses <- zipWithM (\x y -> return $ NeuralNetworkWithLoss x y) children' losses
  -- print len of both children' and losses
  putStrLn $ "Children: " ++ show (length children') ++ " Losses: " ++ show (length losses) ++ " resultVector: " ++ show (length resultVector)
  let sorted = sortOn loss neuralNetworksWithLosses
  let bestChildren = take (maxChildren trainingParameters') sorted
  -- pick best of children, parents and outcome
  best <- pickBestSurvivors oldBatches batch networks (map nn bestChildren) trainingParameters'
  let accuracy = calculateAccuracy batch (head best)
  putStrLn $ "Losses: " ++ show (sum (map loss bestChildren))  ++ " Accuracy: " ++ show accuracy

  return (best, batch)

-- do not use maxIndex as variable name in the following function
calculateAccuracy :: [TrainingData] -> NeuralNetwork -> Double
calculateAccuracy trainingData' nn =
  map (\x -> if maxIndex (feedForward nn (inputs x)) == maxIndex (label x) then 1 else 0) trainingData' & sum & (/ fromIntegral (length trainingData'))

calculateAccuracyWithCounter :: [TrainingData] -> NeuralNetwork -> IO Double
calculateAccuracyWithCounter trainingData' nn =
  let
    accCount :: Int -> [TrainingData] -> Double -> IO Double
    accCount _ [] acc = pure (acc / fromIntegral (length trainingData'))
    accCount counter (x:xs) acc =
      let
        prediction = feedForward nn (inputs x)
        labelIndex = maxIndex (label x)
        predictionIndex = maxIndex prediction
        newAcc = if predictionIndex == labelIndex then acc + 1 else acc
      in do
        when (counter == 1) $ print nn
        when ((counter `mod` 100) == 0) $ putStrLn $ "Iteration " ++ show counter ++ ": Predicted index " ++ show predictionIndex ++ ", Label index " ++ show labelIndex
        accCount (counter + 1) xs newAcc
  in accCount 1 trainingData' 0


testTrainingFirstLine :: IO (Vector Double)
testTrainingFirstLine = do
      trainingData <- trainingDataFromFile "/var/tmp/mnist/mnist_train.csv"
      let layers = [(784, Relu), (100, Relu), (10, SoftMax)]
      nn <- createNeuralNetwork layers
      return $ feedForward nn (inputs $ head trainingData)



trainEpochs :: [TrainingData] -> [NeuralNetwork] -> Int -> TrainingParameters -> [[TrainingData]] -> IO [NeuralNetwork]
trainEpochs trainingData networks epochs trainingparams oldBatches = do
  if epochs == 0 then return networks else do
    putStr $ "Epoch: " ++ show epochs ++ " -- "
    (newNetworks, newBatch) <- trainNetworks oldBatches trainingData networks trainingparams
    let newBatches = take (keepOldBatchesNum trainingparams) (newBatch : oldBatches)
    trainEpochs trainingData newNetworks (epochs - 1) trainingparams newBatches

testTraining :: IO ()
testTraining = do
      trainingData <- trainingDataFromFile "/var/tmp/mnist/mnist_train.csv"
      let layers = [(70, Relu), (70, Relu), (10, SoftMax)]
      nn <- createNeuralNetwork layers
      let trainingParameters = createDefaultTrainingParameters
      trainedNetworks <- trainEpochs trainingData [nn] (epochs trainingParameters) trainingParameters []
      accuracy <- calculateAccuracyWithCounter (take 1000 trainingData) (head trainedNetworks)
      print $ "Accuracy: " ++ show accuracy
      print "Done"