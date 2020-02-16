# carrega pesos da rede
net = googlenet; 

# imagem deve ter 224 x 224 x 3 de tamanho
inputSize = net.Layers(1).InputSize

# mostra imagem
I = read('peppers.png')
figure
imshow(I)

# redimensionando a imagem de acordo com a entrada da rede
I = imresize(i,inputSize(1:2))
figure 
imshow(I)


# classificando a imagem
[label,scores] = classify(net,I);
label

# mostrando resultado 
figure
imshow(I)
title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");

# obtem as 5 melhores probabilidades
[~,idx] = sort(scores,'descend');
idx = idx(5:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

# mostra as 5 melhores probabilidades em um grafico de barras 
figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)
