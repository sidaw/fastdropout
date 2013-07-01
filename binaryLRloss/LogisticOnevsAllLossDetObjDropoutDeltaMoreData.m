function [tnll,g] = LogisticOnevsAllLossDetObjDropoutDeltaMoreData(W,X,Y,ps, Xu, a)
% W(feature,class)
% X(instance,feature)
% Y(instance,class) in 1 of K encoding
K = size(Y,2);
W = reshape(W, [], K);
tnll = 0;
g = zeros(size(W));
nu = size(Xu, 1);
[n,p] = size(X);
pvec = ps * ones(p, 1);
 
for k = 1:K
    w = W(:,k);
    
    y = 2*Y(:,k)-1;
   
    % uncomment next line to dropout bias at a different rate
    % pvec(1) = 0.8;
    alpha = 1;
    
    EXw = X*(w .* pvec);
    X2 = X.*X;
    VarXw = alpha.*(X2) * (w.*w.*pvec.*(1-pvec));
    sigyEx = sigmoid(y.*EXw);
    sigEx = sigmoid(EXw);
    
    EXuw = Xu*(w .* pvec);
    Xu2 = Xu.*Xu;
    VarXuw = alpha.*(Xu2) * (w.*w.*pvec.*(1-pvec));
    sigEXu = sigmoid(EXuw);
    
    s = n/(n+nu*a);
    
    nlldet = sum(-log( sigyEx )) + 0.5.*s.*sum(sigEx.*(1-sigEx).*VarXw)...
    + a*0.5.*s.*sum(sigEXu.*(1-sigEXu).*VarXuw);

    dmu = -y.*(1-sigyEx) + 0.5.*s.*(sigEx.*(1-sigEx).*(1-2*sigEx)).*VarXw;
    dmuu = a*0.5.*s.*(sigEXu.*(1-sigEXu).*(1-2*sigEXu)).*VarXuw;
    dsigma = s.*sigyEx.*(1-sigyEx); % *0.5
    dsigmau = a*s.*sigEXu.*(1-sigEXu); % *0.5
    dEx = pvec .* ((dmu'*X)' + (dmuu'*Xu)');
    dS2 = w.*pvec.*(1-pvec) .* (dsigma'*X2)'; % *2
    dS2u = w.*pvec.*(1-pvec) .* (dsigmau'*Xu2)';

    g(:,k) =  dEx + dS2 + dS2u;
    tnll = tnll + nlldet;
end
g=g(:);