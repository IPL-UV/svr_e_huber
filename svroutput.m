% ---------------------------------------------------------------------------------------------------
%
% svroutput.m
%	
% This function obtains the output of a $\varepsilon$-Huber SVR model given the weigths and bias.
%
% INPUTS:
%	trnX	............. Training set (features in columns, samples in rows).
%	tstX   	............. Test set (features in columns, samples in rows).
%	ker  	............. Label for the kernel type (at this moment only RBF kernel, yker = 'rbf').
%	beta   	............. (Alpha-Alpha^*).
%	bias 	............. Bias of the best model (b).
%	sigmay	............. Best RBF kernel width.
%
% OUTPUTS:
%	tstY	............. Outputs for tstX dataset.
%
% José L. Rojo-Álvarez & Gustavo Camps-Valls                                          
% jlrojo@tsc.uc3m.es, gcamps@uv.es                             
% 2004(c)                                     
%
% ---------------------------------------------------------------------------------------------------

function [tstY] = svroutput(trnX,tstX,ker,beta,bias,sigmay)

H = mysvkernel2(ker,tstX,trnX,sigmay);
tstY = (H*beta +bias);


