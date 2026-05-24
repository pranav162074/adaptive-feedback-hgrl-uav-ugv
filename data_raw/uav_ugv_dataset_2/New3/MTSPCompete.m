clear;close all;
% 初始化参数
startpoints =[ 0,0;0,10;10,0;10,10]; % 旅行商起点
waypoints = [1,0.5;2,0;3,1;5,3;2,4;1,3;0,2;0,8;2,9;8,10;6,0;6,4;9,5;10,3;2,7;6,8;8,7;7,3;]; % 必经点
% 所有点集合，包括三个旅行商的起点
points = [startpoints;waypoints];
n = size(points, 1); % 总点数（包括起点和终点）
n_travelers = size(startpoints,1); % 旅行商数量
dist_matrix = zeros(n, n);
for i = 1:n
    for j = 1:n
        dist_matrix(i, j) = sqrt((points(i, 1) - points(j, 1))^2 + ...
                                 (points(i, 2) - points(j, 2))^2);
    end
end
near_zone=4;%为无人机临近点范围

best_path=[3,15,22,17,18,1,5,6,7,9,10,11,4,21,16,8,20,14,2,13,19,12];
best_path2=[3,15,8,16,22,1,11,10,9,7,2,4,14,18,6,12,20,5,21,19,13,17];



figure;
plot(points(:, 1), points(:, 2), 'ro', 'MarkerSize', 8); % 所有点
hold on;
plot(startpoints(:,1),startpoints(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r'); % 起点
   % 为每个旅行商分配路径
 traveleridx2=find(best_path2<=n_travelers,n_travelers);
    for t = 1:n_travelers
       traveler_path2=travelers_path_func(best_path2 , dist_matrix, n_travelers,near_zone,traveleridx2,t);
        for i = 1:(length(traveler_path2) - 1)
                 plot(points(traveler_path2(i:i+1), 1), points(traveler_path2(i:i+1), 2), '-', 'LineWidth', 2,'Color','green');
        end
        for i = 2:(length(traveler_path2) - 1)
          text(points(traveler_path2(i), 1) + 0.3, points(traveler_path2(i), 2), ['P', num2str(i-1)], 'Color', 'black', 'HorizontalAlignment', 'left');
        end
    end
    
    
    
    traveleridx=find(best_path<=n_travelers,n_travelers);
    for t = 1:n_travelers
       traveler_path=travelers_path_func(best_path , dist_matrix, n_travelers,near_zone,traveleridx,t);
        for i = 1:(length(traveler_path) - 1)
                 plot(points(traveler_path(i:i+1), 1), points(traveler_path(i:i+1), 2), '--', 'LineWidth', 2,'Color',[0.5,0.2,0.8]);
        end
        for i = 2:(length(traveler_path) - 1)
          text(points(traveler_path(i), 1) , points(traveler_path(i), 2)+0.5, ['W', num2str(i-1)], 'Color', 'black', 'HorizontalAlignment', 'left');
        end
    end
% text(points(:, 1), points(:, 2), string(1:n), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
grid on;
title('MTSP');
xlabel('X');
ylabel('Y');