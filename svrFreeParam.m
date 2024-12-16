% ---------------------------------------------------------------------------------------------------
%
% svrFreeParam.m
%
% This function performs the search of the free parameters of the $\varepsilon$-Huber SVR model.
% the function takes the training and test sets and performs the non-exhaustive 
% iterative search strategy was used here. Basically, at each iteration (T iterations), 
% a sequential search on every parameter domain is performed by splitting the range of 
% the parameter in K linearly or logarithmically equally spaced points. Values of T=3 
% and K=20 exhibited good performance in our simulations. 
%
% INPUTS:
%	Xtr		............. Training set (features in columns, samples in rows).
%	Ytr   	............. Actual value to be estimated/predicted in the training set.
%	x_tst 	............. Test set (features in columns, samples in rows).           
%	y_tst 	............. Actual value to be estimated/predicted in the test set.     
%	yker  	............. Label for the kernel type (at this moment only RBF kernel, yker = 'rbf').
%	dibuja	............. Label for plotting. If =1 plots the results in each iteration
%						  see \cite{tr:Camps03a})
%
% OUTPUTS:
%	nsv		............. Number of support vectors.
%	Xsv   	............. Support vector matrix.
%	svs   	............. (Alpha-Alpha^*).
%	bias 	............. Bias of the best model (b).
%	sigmay	............. Best RBF kernel width.
%	C     	............. Best penalization factor.
%	epsil 	............. Best epsilon-insensitivity zone.
%	gam    	............. Best gamma parameter in the $\varepsilon$-Huber cost function.
%
% José L. Rojo-Álvarez & Gustavo Camps-Valls                                          
% jlrojo@tsc.uc3m.es, gcamps@uv.es                             
% 2004(c)                                     
%
% ---------------------------------------------------------------------------------------------------

function [nsv,Xsv,svs,bias,sigmay,C,epsil,gam] = svrFreeParam(Xtr,Ytr,x_tst,y_tst,yker,dibuja)

% Initializing the parameters
epsil     = 1e-7
sigmay    = 1.5
C         = 1e5
gam       = 1e-1

% Initializing the algorithm
Npoints   = 4;           % Nro de puntos para la busqueda
oldError  = 1e10;         % Solo se actualiza el parametro si el error disminuye
lowC      = 3;
upC       = 7;
lowgam    = -10;
upgam     = -5;
loweps    = 0;     
upeps     = 1e-4;
lowsigmay = 1;
upsigmay  = 3;

% Main Loop
error_parcial=[];
warning off
K=3;	% iterations
for vuelta=1:K

	disp(['Iteration ' num2str(vuelta) ' of ' num2str(K)])
	
	% Search in gam
    err=zeros(Npoints,1);
    ggam=logspace(lowgam,upgam,Npoints);
    for i=1:Npoints
        [nsv,svs,bias] = mysvr(Xtr,Ytr,epsil,C,ggam(i),sigmay,yker);        
        ypredtest = svroutput(Xtr,x_tst,yker,svs,bias,sigmay);
        err(i) = mean((ypredtest-y_tst).^2);
    end
    if dibuja,
        subplot(321); loglog(ggam,err); axis tight; 
        xlabel('\gamma'),ylabel('MSE'),grid, drawnow
    end
    [kk,m] = min(err); 
    if kk<oldError,
        gam = ggam(m);  oldError=kk;
    end

    % Search in C
    err=zeros(Npoints,1);
    CC=logspace(lowC,upC,Npoints);
    for i=1:Npoints
        [nsv,svs,bias] = mysvr(Xtr,Ytr,epsil,CC(i),gam,sigmay,yker);        
        ypredtest = svroutput(Xtr,x_tst,yker,svs,bias,sigmay);
        err(i) = mean((ypredtest-y_tst).^2);
    end
    if dibuja,
        subplot(322); loglog(CC,err); axis tight; 
        xlabel('C'),ylabel('MSE'),grid, drawnow
    end
    [kk,m] = min(err); 
    if kk<oldError,
        C = CC(m);    oldError=kk;
    end

    % Search in epsilon
    err=zeros(Npoints,1);
    eepsil=linspace(loweps,upeps,Npoints);
    for i=1:Npoints
        [nsv,svs,bias] = mysvr(Xtr,Ytr,eepsil(i),C,gam,sigmay,yker);
        ypredtest = svroutput(Xtr,x_tst,yker,svs,bias,sigmay);
        err(i) = mean((ypredtest-y_tst).^2);
    end
    if dibuja,
        subplot(323); 
        plot(eepsil,err); axis tight; xlabel('\epsilon'),ylabel('MSE'), grid, drawnow
    end
    [kk,m] = min(err); 
    if kk<oldError,
        epsil = eepsil(m);    oldError = kk;
    end             
    
    % Search in sigmay
    err=zeros(Npoints,1);
    ssigmay=logspace(log10(lowsigmay),log10(upsigmay),Npoints);
    for i=1:Npoints
        [nsv,svs,bias] = mysvr(Xtr,Ytr,epsil,C,gam,ssigmay(i),yker);
        ypredtest = svroutput(Xtr,x_tst,yker,svs,bias,ssigmay(i));
        err(i) = mean((ypredtest-y_tst).^2);
    end
    if dibuja,
        subplot(324); loglog(ssigmay,err); axis tight; 
        xlabel('\sigma_y'),ylabel('MSE'), grid, drawnow
    end
    [kk,m] = min(err); 
    if kk<oldError,
        m2 =max(find(err==kk)); sigmay = ssigmay(m2);        
        oldError = kk;
    end
    
    % Partial error
    error_parcial = [error_parcial oldError];
    if dibuja
        subplot(325);  plot(error_parcial,'.-.'); axis tight;
        xlabel('error'),ylabel('MSE'), grid, drawnow;
    end
    
end

warning on
[nsv,svs,bias] = mysvr(Xtr,Ytr,epsil,C,gam,sigmay,yker);

tol=1e-7;
aux=find(abs(svs)>=tol);
svs =svs(aux);
Xsv =Xtr(aux,:);
nsv=length(svs);
