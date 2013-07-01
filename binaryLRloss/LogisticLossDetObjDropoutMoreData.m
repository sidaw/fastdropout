function [nll,g] = LogisticLossDetObjDropoutMoreData(w,X,y,ps, X_u, disccoef)
[n,p] = size(X);
[nu,~] = size(X_u);
pvec = ps * ones(p, 1);
% uncomment next line to never drop out the bias
% pvec(1) = 0.8;
alpha = 1;
y01 = (y+1)./2;

[EXw, X2, c, sigcEx] =...
    getIntermediates(w, X, ps, alpha);

[EXw_u, X2_u, c_u, sigcEx_u] =...
    getIntermediates(w, X_u, ps, alpha);
kappa = 0.125*pi;


a = n/(n+disccoef*nu);
b = n*disccoef/(n+disccoef*nu);

psi_sup = -sum( log(1-sigcEx )./c );
psi_unsup = -sum( log(1-sigcEx_u )./c_u );
nlldet = -sum(y01.*(EXw)) + a*psi_sup + b*psi_unsup;


% dmu_sup = sigcEx;
% dmu_unsup = sigcEx_u;
dsigma_sup = alpha.*kappa.*c.*( EXw.*(1-sigcEx).*c...
    - log(sigcEx)); % * 0.5 cancels with YYY
dsigma_unsup = alpha.*kappa.*c_u.*( EXw_u.*(1-sigcEx_u).*c_u...
    - log(sigcEx_u)); % * 0.5 cancels with YYY


dEx = pvec .* ( (-y01 + a*sigcEx)'*X + b*sigcEx_u'*X_u)';
dS2 = w.*pvec.*(1-pvec) .* ...
    (a*dsigma_sup'*X2 + b*dsigma_unsup'*X2_u)'; % *2 YYY
g =  dEx + dS2;
    

% nlldetnaive = -sum(log( sigmoid(y.*c.*EXw) ));
% b = [nllgauss nlldet nlldetnaive ];
nll = nlldet;
% disp(b)
% nll = sum(mean( log1plusexp(-yXw), 2));
end


function [EXw, X2, c, sigcEx] = getIntermediates(w, X, pvec, alpha)
EXw = X*(w .* pvec);
X2 = X.*X;
SigmaXw2 = alpha.*(X2) * (w.*w.*pvec.*(1-pvec));
kappa = 0.125*pi;
one_plus_sigma = (1 + kappa.*SigmaXw2);
c = 1./sqrt(one_plus_sigma);
sigcEx = sigmoid(c.*EXw);
end

