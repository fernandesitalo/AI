%~ # # # classificar de qual vinicula o vinho veio (de um total de 3)
%~ # # # de acordo com 13 atributos:
%~ # Alcohol
%~ # Malic acid
%~ # Ash
%~ # Alkalinity of ash
%~ # Magnesium
%~ # Total phenols
%~ # Flavonoids
%~ # Nonflavonoid phenols
%~ # Proanthocyanidins
%~ # Color intensity
%~ # Hue
%~ # OD280/OD315 of diluted wines
%~ # Proline

%~ # matriz de entrada x : contem 13 linhas x 178 colunas  .: 178 vinhos conhecidos
%~ # matriz de entrada t : contem 3 linhas x 178 colunas .: 178 respostas para entrada
%~ # # # se o vinho eh da vinivula 1 somente a linha 1 daquela coluna vai ter o numero igual a 1


[x,t] = wine_dataset;

%~ # 1 camada com 10 neuronioss
net = patternnet(10); 
view(net)

[net,tr] = train(net,x,t);
nntraintool

%~ # plota grafico de edesempenho do treinamento...
plotperform(tr)


%~ #testando a rede
testX = x(:,tr.testInd);
testT = t(:,tr.testInd);

testY = net(testX);
testIndices = vec2ind(testY)

%~ #plota matriz de confusão: 
%~ # # # a diagonal principal eh o quanto a classa foi classificada corretamente
%~ # # # as linhas são as classes preditas e as colunas sao as classes reais
%~ # # # usada para validação e analise da rede.
plotconfusion(testT,testY)


%~ # c eh a porcentagem de calc incorreto, cm e a matriz de confusao
[c,cm] = confusion(testT,testY)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

%~ # plota matriz ROC
%~ # # # a diagonal segundaria representa um classificador aleatorio
%~ # # # a baixo da diagonal segundaria o classificador e considerado ruim pois um aleatorio e melhor
%~ # # # a cima da diagonal sefgundaria o classificador é considerador bom.
plotroc(testT,testY)

