%~ são 13 dados de entrada:
%~ #Age (years)
%~ #Weight (lbs)
%~ #Height (inches)
%~ #Neck circumference (cm)
%~ #Chest circumference (cm)
%~ #Abdomen circumference (cm)
%~ #Hip circumference (cm)
%~ #Thigh circumference (cm)
%~ #Knee circumference (cm)
%~ #Ankle circumference (cm)
%~ #Biceps (extended) circumference (cm)
%~ #Forearm circumference (cm)
%~ #Wrist circumference (cm)

%~ são 252 dados de entrada e uma saida
matriz x tem 
[X,T] = bodyfat_dataset;


%~ #
net = fitnet(15);
view(net)

[net,tr] = train(net,X,T);
nntraintool


%~ #
plotperform(tr)


%~ #
testX = X(:,tr.testInd);
testT = T(:,tr.testInd);

testY = net(testX);

perf = mse(net,testT,testY)

%~ #
Y = net(X);

plotregression(T,Y)

%~ #
e = T - Y;

ploterrhist(e)
