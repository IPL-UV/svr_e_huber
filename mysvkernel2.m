function H = mysvkernel(ker,x1,x2,sigmay)

% ---------------------------------------------------------------------------------------------------
% SVKERNEL kernel for Support Vector Methods
%
%  Usage: k = svkernel(ker,u,v)
%
%  Parameters: ker - kernel type
%              u,v - kernel arguments
%
%  Values for ker: 'linear'  -
%                  'poly'    - p1 is degree of polynomial
%                  'rbf'     - p1 is width of rbfs (sigma)
%                  'sigmoid' - p1 is scale, p2 is offset
%                  'spline'  -
%                  'bspline' - p1 is degree of bspline
%                  'fourier' - p1 is degree
%                  'erfb'    - p1 is width of rbfs (sigma)
%                  'anova'   - p1 is max order of terms
%              
%  Author: Steve Gunn (srg@ecs.soton.ac.uk)
% ---------------------------------------------------------------------------------------------------

% ---------------------------------------------------------------------------------------------------
% Version for fast matrix form of RBF kernel
%
% Code modified by José L. Rojo-Álvarez & Gustavo Camps-Valls                                          
% jlrojo@tsc.uc3m.es, gcamps@uv.es                             
% 2004(c)                                     
%
% ---------------------------------------------------------------------------------------------------

if strcmp(ker,'rbf')
    
    if(size(x1,2)==1)
        N1=size(x1,1);
        N2=size(x2,1);
        H = zeros(N1,N2);
        for i=1:N1
            H(i,:) = (exp(-(1/2/(sigmay^2))*(x2-ones(N2,1)*x1(i,:))'.*(x2-ones(N2,1)*x1(i,:))'));
        end
    else
        N1=size(x1,1);
        N2=size(x2,1);
        H = zeros(N1,N2);
        for i=1:N1
            H(i,:) = exp(-(1/2/(sigmay^2))*sum((x2-ones(N2,1)*x1(i,:))'.*(x2-ones(N2,1)*x1(i,:))'));
        end
    end
    
elseif strcmp(ker,'semirbf')
    
    global p
    N1=size(x1,1);
    N2=size(x2,1);
    
    % Componentes no parametricas
    xm1 = x1(:,1:p);
    xm2 = x2(:,1:p);
    M = zeros(N1,N2);
    for i=1:N1
        M(i,:) =  exp(-(1/2/(sigmay^2))*...
            sum((xm2-ones(N2,1)*xm1(i,:))'.*(xm2-ones(N2,1)*xm1(i,:))'));
    end
    
    % Componentes parametricas
    xc1 = x1(:,p+1:end);
    xc2 = x2(:,p+1:end);
    C = xc1 * xc2';
    
    H = M + C;
end