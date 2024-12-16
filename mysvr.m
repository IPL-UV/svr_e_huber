% ---------------------------------------------------------------------------------------------------
%
% mysvr.m
%	
% This function trains a $\varepsilon$-Huber SVR model with a fixed set of free parameters.
%
% INPUTS:
%	x		............. Training set (features in columns, samples in rows).
%	y   	............. Actual value to be estimated/predicted in the training set.
%	e 		............. Best epsilon-insensitivity zone.
%	C     	............. Best penalization factor.
%	gam    	............. Best gamma parameter in the $\varepsilon$-Huber cost function.
%	sigmay	............. Best RBF kernel width.
%	ker  	............. Label for the kernel type (at this moment only RBF kernel, yker = 'rbf').
%
% OUTPUTS:
%	nsv		............. Number of support vectors.
%	w   	............. (Alpha-Alpha^*).
%	bias 	............. Bias of the best model (b).
%	ypred   ............. Predictions for desired y
%
% José L. Rojo-Álvarez & Gustavo Camps-Valls                                          
% jlrojo@tsc.uc3m.es, gcamps@uv.es                             
% 2004(c)                                     
%
% ---------------------------------------------------------------------------------------------------


function [nsv,w,bias, ypred] = mysvr(x,y,e,C,gam,sigmay,ker)

% Initialization
N = length(y);

% Construct kernel matrix and alpha vector  
H = mysvkernel2(ker,x,x,sigmay);
alfa=zeros(N,1);   % Incognitus
alfabis=zeros(N,1);
w = zeros(N,1);
bias=0;
maxiter=10;
t=.95;               % Update lambda^(*)
pp=zeros(maxiter,1);

for i=1:maxiter
    
    % Compute errors
    ypred=H*w;
    er=y-ypred;
    pp(i)=norm(er);
    % Compute alphas
    [alfa,alfabis]=misalfas(er,e,C,gam);
    % Compute lambdas
    [landa_new,landabis_new]=mislandas(alfa,alfabis,er,e);
    if (i==1)
        landa=landa_new;    
        landabis=landabis_new;
        landa_old=landa;
        landabis_old=landabis;
    else
        landa=t*landa_new+(1-t)*landa_old;
        landabis=t*landabis_new+(1-t)*landabis_old;
        landa_old=landa;
        landabis_old=landabis;
    end
    aux=find( (landa+landabis)~=0 );
    An= eye(N) + diag(landa+landabis)*H;
    Sin = diag(landa+landabis)*y - ...
        diag(landa-landabis)*(e*ones(N,1));
    w=inv(An)*(Sin);
end

aux=find(abs(w)>1e-7);
nsv=length(aux);

% ==========================================
% ========= Auxiliary functions ============
% ==========================================

function [alfa,alfabis]=misalfas(er,e,C,gam)

N0=length(er);
alfa=zeros(N0,1);
alfabis=zeros(N0,1);
ec=e+gam*C;

S1 = find( (er>=e)&(er<=ec) );
S1b= find( (er<=-e)&(er>=-ec) );
S2 = find( (er>ec) );
S2b= find( (er<-ec) );

alfa(S1) = 1/gam*(er(S1)-e);
alfabis(S1b)= 1/gam*(-er(S1b)-e);
alfa(S2) = C;
alfabis(S2b)= C;


function [landa,landabis]=mislandas(alfa,alfabis,er,e);

N0=length(er);
erb=er;
er=er-e;
erb=-erb-e;
landa=zeros(1,N0);
landabis=zeros(1,N0);

V1=find(abs(alfa-alfabis)<1e-10);
V1=setdiff(1:N0,V1);

landa(V1) = 2*alfa(V1)./er(V1);
landabis(V1) = 2*alfabis(V1)./erb(V1);
