 function [optimalPath,fitnessFunc]=Gene(x0, y0, xf, yf, waypoints)

% clear; close all;
% %初始条件
% x0 = 0; y0 = 0; % 起点
% xf = 10; yf = 10; % 终点
% waypoints = [4, 5; 9, 6; 9, 10; 6, 6; 5, 9]; % 必经点

tic();
% 所有点集合
points = [x0, y0; waypoints; xf, yf];


% 距离矩阵计算   
n = size(points, 1); % 点的总数
distMatrix = zeros(n, n);
for i = 1:n
    for j = 1:n
        distMatrix(i, j) = norm(points(i, :) - points(j, :)); % 欧几里得距离
    end
end

% 求解TSP问题（起点固定为1，终点固定为最后一个点）
idx_start = 1; % 起点索引
idx_end = n; % 终点索引

% 使用遗传算法求解TSP问题
fitnessFunc = @(order) tspFitness(order, distMatrix, idx_start, idx_end);
nWaypoints = n - 2; % 必经点数（不包括起点和终点）

% 遗传算法参数设置
%opts = optimoptions('ga', 'Display', 'iter', 'UseParallel', true);
opts = optimoptions('ga', 'Display', 'iter', 'UseParallel', false);
% 采用全排列的个体表示（确保路径中点顺序唯一）
order = ga(fitnessFunc, nWaypoints, [], [], [], [], ...
   ones(1, nWaypoints), nWaypoints * ones(1, nWaypoints), [], 1:nWaypoints, opts);

% 将结果转换为实际路径
optimalOrder = [idx_start, order + 1, idx_end]; % 起点 → 必经点 → 终点
optimalPath = points(optimalOrder, :);

% 绘制最优路径
% figure;
% plot(points(:, 1), points(:, 2), 'bo', 'MarkerSize', 8); % 所有点
% hold on;
% plot(optimalPath(:, 1), optimalPath(:, 2), 'r-', 'LineWidth', 2); % 最优路径
% text(points(:, 1), points(:, 2), string(1:n), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
% grid on;
% title('最优路径规划');
% xlabel('X坐标');
% ylabel('Y坐标');
% hold off;

% 显示最优路径点序列
disp('最优点访问顺序:');
disp(optimalOrder);
t=toc();
disp(['t=',num2str(t)])
  end
% --- TSP 适应度函数 ---
function totalDistance = tspFitness(order, distMatrix, idx_start, idx_end)
    % 检查路径是否包含重复点
    if length(unique(order)) ~= length(order)
        totalDistance = inf; % 如果路径包含重复点，返回极大的适应度值
        return;
    end
    
    % 将顺序加上起点和终点
    fullOrder = [idx_start, order + 1, idx_end]; % 加入起点和终点
    
    totalDistance = 0;
    for i = 1:length(fullOrder) - 1
        totalDistance = totalDistance + distMatrix(fullOrder(i), fullOrder(i + 1)); % 计算路径长度
    end
 end


