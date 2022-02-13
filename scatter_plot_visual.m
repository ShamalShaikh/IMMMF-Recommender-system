hold on
scatter(X(1,:), randi(1000, 1, m),'green','filled');
xline(theta(1,:),'b');

% color = ['yellow', 'yellow', 'yellow', 'yellow', 'yellow'];

% scatter(X(1,:)<=Theta, randi(1000, 1, m),color(1),'filled');
xline(Theta(1,1),'--b');

 xline(rightBoundary(1,1),'--b');
    xline(leftBoundary(1,1),'--b');

    %     scatter((Ynew(1,:)==l+1), randi(1000, 1, size(Ynew(1,:),2)),color(l+1),'filled');

    xline(Theta(1,1),'--b');
% scatter(X(1,:)>=Theta, randi(1000, 1, m),color(L),'filled');

hold off