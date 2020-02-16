%~ # # # classificacao dos caranguejos em macho e femea de acordo com 6 caracteristicas:

%~ # species
%~ # frontallip
%~ # rearwidth
%~ # length
%~ # width
%~ # and 
%~ # depth

%~ # matriz x: e a entrada de dados 6 linhas x 200 colunas
%~ # matriz t: e a classificacao dos respectivos dados 2 linhas x 200 colunas
[x,t] = crab_dataset;


%~ # monta rede com 1 camada olcuta com 10 neuronios
net = patternnet(10);
view(net)


%~ # trienamento da rede
[net,tr] = train(net,x,t);
nntraintool

%~ # avaliar o desempenho do treinamento e validação da rede com um grafico de entropia
plotperform(tr)

%~ #testando o classificador
testX = x (:, tr.testInd); 
testT = t (:, tr.testInd); 

testY = líquido (testX); 
testIndices = vec2ind (testY)

%~ #plotar matriz de confusão
plotconfusion (testT, testY)

[c, cm] = confusão (testT, testY) 

fprintf ( 'Classificação correta da porcentagem:% f %% \ n' , 100 * (1-c)); 
fprintf ( 'Classificação incorreta da porcentagem:% f %% \ n' , 100 * c);

%~ # plotar grafico de ROC
plotroc (testT, testY)
