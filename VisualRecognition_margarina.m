clear;
%faz string do caminho
base_path = fullfile('base_de_dados');

% Define banco de imagens e dimensão das imagens.
imds = imageDatastore(base_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
inputSize = [45 246];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);

% Define o numero de arquivos para treino.
labelCount = countEachLabel(imds)
numTrainFiles = 47;
% Separa as imagens para treino e validação
[imdsTrain,imdsValidation] = splitEachLabel(imds, numTrainFiles,'randomize');
numTestFiles = 16;
[imdsTest, imdsValidation] = splitEachLabel(imdsValidation, numTestFiles,'randomize');

% Define a topologia da rede
layers = [
  imageInputLayer([inputSize 3])% Layer 1 - Apenas especifica a dimensão das imagens.
  
  convolution2dLayer(3,8,'Padding','same')% Layer 2 - Definir o tamanho dos filtros, 3x3 -- PARA APLICAÇÃO DE FILTROS NA ENTRADA.
  batchNormalizationLayer                 % o numero de neuronios, 8, que define o numero de features.
  reluLayer								  % funcão de ativação não linear 
  
  maxPooling2dLayer(2,'Stride',2)% Layer 3 - Reduz os tamanho das features de entreda da camada. 
  convolution2dLayer(3,16,'Padding','same')% Layer 4 - mesma que a 2!!!!!
  batchNormalizationLayer
  reluLayer								  % funcão de ativação não linear 
  
  maxPooling2dLayer(2,'Stride',2)% Layer 5 - mesma que a 3!!!
  convolution2dLayer(3,32,'Padding','same')% Layer 6 - mesma que a 4!!!!!
  batchNormalizationLayer
  reluLayer								 % funcão de ativação não linear 
  
  fullyConnectedLayer(2)% Layer 7 - Conecta com todos os neurorions da camada anterior, 2 significa que tera duas saidas.
  softmaxLayer
  classificationLayer
 ];

% Define as opções de treino
options = trainingOptions('sgdm', ...
  'InitialLearnRate',0.01, ...
  'MaxEpochs',50, ...
  'Shuffle','every-epoch', ... %% Todas as epocas, as imagens são embaralhadas!!
  'ValidationData',imdsValidation, ...
  'ValidationFrequency',30, ...
  'Verbose', false, 'Plots','training-progress');

[net, tr] = trainNetwork(imdsTrain, layers, options);

YPred = classify(net, imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

% Plot dos resultados
predicted_labels = classify(net, imdsTest);

expected_labels = imdsTest.Labels;

figure
plotconfusion(expected_labels, predicted_labels); % Matriz de confusão

figure
plotroc(double(expected_labels'), double(predicted_labels')); % Curva ROC
