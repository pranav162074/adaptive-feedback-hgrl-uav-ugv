 function [optimalPath,fitnessFunc]=Gene2(x0, y0, xf, yf, waypoints,param)
tic();
% 所有点集合
points = [x0, y0;waypoints; xf, yf];
n = size(points, 1); % 点的总数
global distMatrix ;%存储各点间距离长度
distMatrix = zeros(n, n);
% 求解TSP问题（起点固定为1，终点固定为最后一个点）
idx_start = 1; % 起点索引
idx_end = n; % 终点索引
% 使用遗传算法求解TSP问题
fitnessFunc = @(order) tspFitness(order,points, idx_start, idx_end,param);
nWaypoints = n - 2; % 必经点数（不包括起点和终点）
% 遗传算法参数设置
opts = optimoptions('ga', 'Display', 'iter', 'UseParallel', false);
% 采用全排列的个体表示（确保路径中点顺序唯一）
order = ga(fitnessFunc, nWaypoints, [], [], [], [], ...
   ones(1, nWaypoints), nWaypoints * ones(1, nWaypoints), [], 1:nWaypoints, opts);

% 将结果转换为实际路径
optimalOrder = [idx_start, order + 1, idx_end]; % 起点 → 必经点 → 终点
optimalPath = points(optimalOrder, :);
% 显示最优路径点序列
disp('最优点访问顺序:');
disp(optimalOrder);
t=toc();
disp(['t=',num2str(t)])
 end
% --- TSP 适应度函数 ---
function totalDistance = tspFitness(order,points, idx_start, idx_end,param)
    % 检查路径是否包含重复点
    global distMatrix ;
    if length(unique(order)) ~= length(order)
        totalDistance = inf; % 如果路径包含重复点，返回极大的适应度值
        return;
    end
    % 将顺序加上起点和终点
    fullOrder = [idx_start, order + 1, idx_end]; % 加入起点和终点
    current_points=points(fullOrder,:);
    num_points=size(current_points,1);
     totalDistance = 0;
    for i = 1:num_points-1
     x_start = current_points(i, 1);
         y_start = current_points(i, 2);
         x_end = current_points(i+1, 1);
         y_end = current_points(i+1, 2);
          if distMatrix(fullOrder(i),fullOrder(i+1))~=0
             path_length=distMatrix(fullOrder(i),fullOrder(i+1)); 
         else
     if ~check_obstacle(x_start, y_start, x_end, y_end,100,param.obstacles)
        % 如果没有障碍物，直接连接这两个点
        x_path = [x_start; x_end];
        y_path = [y_start; y_end];
    else
         p_start=[x_start;y_start];
                 p_end=[x_end; y_end];
                result = RRTstar12(param, p_start, p_end);  % 正确的调用
                 x_path = transpose(result.refinedP(1,:));%转置
              y_path = transpose(result.refinedP(2,:));     
     end
      path_length = 0;
    for j = 1:length(x_path) - 1
        path_length = path_length + norm([x_path(j+1), y_path(j+1)] - [x_path(j), y_path(j)]);
    end
    distMatrix(fullOrder(i),fullOrder(i+1))=path_length;
    distMatrix(fullOrder(i+1),fullOrder(i))=path_length;
          end
        totalDistance = totalDistance + path_length; % 计算路径长度
    end
end


