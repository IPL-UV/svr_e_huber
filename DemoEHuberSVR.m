
% ---------------------------------------------------------------------------------------------------
% This script shows the performance of the $\varepsilon$-Huber SVR model with the MERIS 
% dataset included in "meris_data.mat". The regression problem consits of
% estimating oceanic chlorophyll content from simulated MERIS spectral
% data. More information in:
%
%  Robust support vector regression for biophysical variable estimation from remotely sensed images 
%  Camps-Valls, G. and Bruzzone, L. and Rojo-Álvarez, J.L. and Melgani, F.
%  IEEE Geoscience and Remote Sensing Letters 3 (3):339-343, 2006 
%
%  A more detailed theoretical analysis of the proposed cost function was
%  published in:
%
%    A unified SVM framework for signal estimation
%    Rojo-Álvarez, J.L. and Martínez-Ramón, M. and Muñoz-Marí, J. and Camps-Valls, G. 
%    Digital Signal Processing: A Review Journal 26 (1):1-20, 2014
%
% Jose L. Rojo-Alvarez & Gustavo Camps-Valls                                          
% jlrojo@tsc.uc3m.es, gcamps@uv.es                             
% 2004(c)                                     
% ---------------------------------------------------------------------------------------------------

clear;clc;

% ---------------------------------------------------------------------------------------------------
% Load data and assign training/test/validation sets (SamplesXX) as long as the 
% corresponding desired outputs (LabelsXX).
% ---------------------------------------------------------------------------------------------------
load('meris_data.mat')
SamplesT  = Train1xy(:,1:end-1); LabelsT  = Train1xy(:,end);
SamplesV  = Train2xy(:,1:end-1); LabelsV  = Train2xy(:,end);
SamplesTE = Testxy(:,1:end-1);   LabelsTE = Testxy(:,end);

% ---------------------------------------------------------------------------------------------------
% Flags:
% ---------------------------------------------------------------------------------------------------
dibuja = 1;		% Plot a follow-up pictures through the automating tuning of the free parameters.
ker = 'rbf';	% Define kernel. At present, only the RBF kernel is possible.
TUNE = 1; 		% Flag variable to allow (if TUNE=1) automating tuning of the free 
		  		% parameters (C,gamma,delta,epsilon) through split-and-selection 
		  		% procedure (see \cite{tr:Camps03a}).
% ---------------------------------------------------------------------------------------------------

if TUNE
	% ---------------------------------------------------------------------------------------------------
	% TUNING THE FREE PARAMETERS. This piece of source code tunes the free parameters by means 
	% of automatic split-and-selection procedure (see \cite{tr:Camps03a}). Please note that this may 
	% be very time consuming, depending on the PC. For this purpose, we settled a not-so-exhaustive 
	% search by fixing the number of iterations in the search procedure to K=3, and the number of 
	% splitting points to N=20.
	% ---------------------------------------------------------------------------------------------------
	 [nsv,Xsv,svs,bias,sigmay,C,epsil,gam] = svrFreeParam(SamplesT,LabelsT,SamplesV,LabelsV,'rbf',dibuja);
	 PreLabelsT = svroutput(Xsv,SamplesT,ker,svs,bias,sigmay);	
	 PreLabelsV = svroutput(Xsv,SamplesV,ker,svs,bias,sigmay);  
	 PreLabelsTE = svroutput(Xsv,SamplesTE,ker,svs,bias,sigmay);

	 [ME,RMSE,ABSE,r] = ComputeResults(LabelsT,PreLabelsT)
	 [ME,RMSE,ABSE,r] = ComputeResults(LabelsV,PreLabelsV)
	 [ME,RMSE,ABSE,r] = ComputeResults(LabelsTE,PreLabelsTE)

else

	% ---------------------------------------------------------------------------------------------------
	% THE BEST SET OF FREE PARAMETERS FOR THE "MERIS DATASET". After an exhaustive search,
	% these are the best subset of free parameters.
	% ---------------------------------------------------------------------------------------------------
	sigmay  =  3
	C       =  1e+8
	epsil   =  1e-003
	gam     =  1e-010
	
	[nsv,svs,bias] = mysvr(SamplesT,LabelsT,epsil,C,gam,sigmay,ker);
	tol = 1e-7;
	aux = find(abs(svs)>=tol); 
	svs = svs(aux);
	Xsv = SamplesT(aux,:);
	nsv = length(svs);
	
	PreLabelsT  = svroutput(Xsv,SamplesT,ker,svs,bias,sigmay);	
	PreLabelsV  = svroutput(Xsv,SamplesV,ker,svs,bias,sigmay);  
	PreLabelsTE = svroutput(Xsv,SamplesTE,ker,svs,bias,sigmay); 
	
	[ME,RMSE,ABSE,r] = ComputeResults(LabelsTE,PreLabelsTE)
	
	errors = LabelsTE-PreLabelsTE;
	h=hist(errors,1000);
	hist(errors,1000);
	hold on
	plot(epsil*ones(1,max(h)),1:max(h),':k'),
	plot(-epsil*ones(1,max(h)),1:max(h),':k'),
	plot(gam*C*ones(1,max(h)),1:max(h),':k'),
	plot(-gam*C*ones(1,max(h)),1:max(h),':k'),
	
end;	
