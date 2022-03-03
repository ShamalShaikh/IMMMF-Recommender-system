function [Ynew] = refineData(X, theta, spreadPerFromCent)

X = full(X);
[n,m] = size(X);
Ynew = zeros(n,m);
L = size(theta,2) + 1;

% scatter_plot_visual()

%% only refining 2:4 ratings
%{
Theta = theta(:,1)*ones(1,m);
Ynew(X<=Theta) = 1;

for l=1:L-2
    %centroid(:,l) = sum(X.* (y==l), 2)./ ( sum(y==l,2) + eps);
    
    dist = abs(theta(:,l+1)-theta(:,l));
    
    leftBoundary = theta(:,l) + dist*spreadPerFromCent/100;
    rightBoundary = theta(:,l+1) - dist*spreadPerFromCent/100;
    
    Ynew(X>=leftBoundary & X<=rightBoundary) = l + 1;
end

Theta = theta(:,4)*ones(1,m);
Ynew(X>=Theta) = L;
%}
%% refining all ratings
%
averageThetaD = zeros(n,1);

for l=1:L-2
     averageThetaD = averageThetaD + abs(theta(:,l+1)-theta(:,l));
end
averageThetaD = averageThetaD./(L-2);

Theta = (theta(:,1) - averageThetaD*spreadPerFromCent/100)*ones(1,m); %shift boundary for rating 1
%Ynew(X<=Theta & Y == 1) = 1;
Ynew(X<=Theta) = 1;
for l=1:L-2
    %centroid(:,l) = sum(X.* (Y==l), 2)./ ( sum(y==l,2) + eps);
    
%     dist = abs(theta(:,l+1)-theta(:,l));
%     leftBoundary = (theta(:,l) + dist*spreadPerFromCent/100)*ones(1,m);
%     rightBoundary = (theta(:,l+1) - dist*spreadPerFromCent/100)*ones(1,m);

    leftBoundary = (theta(:,l) + averageThetaD * spreadPerFromCent/100)*ones(1,m);
    rightBoundary = (theta(:,l+1) - averageThetaD * spreadPerFromCent/100)*ones(1,m);

    %Ynew(X>=leftBoundary & X<=rightBoundary & Y== (l + 1)) = l + 1;
    Ynew( (X>=leftBoundary) & (X<=rightBoundary) & (Ynew==0) ) = l + 1;

end

Theta = (theta(:,4) + averageThetaD*spreadPerFromCent/100 )*ones(1,m);

%Ynew(X>=Theta & Y == 5) = L;
Ynew( (X>=Theta) & (Ynew==0)) = L;

%}
end