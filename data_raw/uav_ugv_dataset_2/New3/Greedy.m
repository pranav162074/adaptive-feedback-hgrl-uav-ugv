clear; close all;

% 初始条件
tic();
x0 = 0; y0 = 0; % 起点

%case1：不返回起点
% xf = 10; yf = 10; % 终点
% waypoints = [ 4,5 ;9,6; 9,10 ;6,6; 5,9]; % 必经点
% param.obstacles=[2,1,2,3;
%   3,3,7,2;
%   2,2,2,7;
%   2,2,8,8;];

% %case1:返回起点
% xf = 0; yf = 0; % 终点
% waypoints = [2,8;9,9;8,2;]; % 必经点
% param.obstacles=[4,4,5,5;];
 
% %case2:
% xf = 0; yf = 0; % 终点
% waypoints = [ 4,5 ;9,6; 9,10 ;6,6; 5,9;10,3;6,1]; % 必经点
% param.obstacles=[2,1,2,3;
%   2,2,7,3;
%   2,2,2,7;
%   2,2,8,8;];

% %case3：
% xf = 0; yf = 0; % 终点
% waypoints = [ 3,5 ;9,6; 9,10 ;6,6.5; 5,9;10,3;6,1]; % 必经点
% param.obstacles=[1,1.5,2,3;
%   1.5,1,7,3;
%   1,1.5,2,7;
%   1.5,1,8,8;
%   1,1,3,7;
%   1,2,7,4;
%   0.5,1.5,4,0;
%   1,1,5,5;
%   0.5,0.5,5,7;
%   1,1,7,9;
%   ];

% %case4:
% xf = 0; yf = 0; % 终点
% waypoints = [ 2,7;2,2;4,4;6,2;9,2;9,6;7,9;3.5,1;10,10 ]; % 必经点
% param.obstacles=[1,1,1,2.5;
%     1,1,0.5,4.5;
%     1,1,1,8;
%     0.5,0.5,2.5,1;
%     1,1,5,1;
%     1,1,4,2;
%     1,1,3.5,5;
%     1,1,5.5,9;
%     1,1,7.5,2;
%     2,2,7,6;
%     1,1,9.5,1;
%     1,1,9.5,4;
%     1,1,9,9;
%     ];

%case5:
xf = 0; yf = 0; % 终点
waypoints = [ 1,6;4,8;10,10;9,2;4,1.5;6,4 ]; % 必经点 
param.obstacles=[2,2,1.5,4;
    1,1,2,7.5;
    1,1,3,1.5;
    1,1,5,3;
    1.5,1.5,5,6.5;
    1,1,6.5,1;
    1,1,8,2;
    1,1,7.5,5;
    2,2,8.5,8;
    ];

H = 10; % 每段的时间步长
V = 1; % 无人机速度
u_max = 0.2; % 最大加速度
stop_time = 0; % 每个必经点的停留时间

param.searchFeild = [-3,12,-3,12];   % 搜索区域
param.threshold = 1;   % d_{min}
param.theta = 5*pi/6;    % 论文中theta取150度
param.resolution = 10;   % 决定在路径上多远取一个点，以判断路径是否避开障碍
param.maxNodes = 5000;  %算法在停止之前会生成的最大节点数，最大迭代次数
param.step_size = 1;  %新节点向采样点生长时的最大步长
param.neighbourhood = 1; % 设置neighbourhood的半径r
param.random_seed =40;
param.u_max = 0.2;    %最大曲率  


% 障碍参数（椭圆）
num_obs=size(param.obstacles,1);%障碍数
for i=1:num_obs
a(i)= param.obstacles(i,1); % 长轴
b(i) = param.obstacles(i,2); % 短轴
c (i)= param.obstacles(i,3); % 椭圆中心x坐标
d (i)= param.obstacles(i,4); % 椭圆中心y坐标
end

% 使用贪婪算法优化必经点顺序
%[sorted_waypoints, total_distance] = greedy_order(x0, y0, xf, yf, waypoints);
%遗传算法
[sorted_waypoints, total_distance] = Gene(x0, y0, xf, yf, waypoints);

disp('优化后的必经点顺序:');
disp(sorted_waypoints);

% 初始化路径长度和时间
total_path_length = 0;

% 路径规划处理多个路径段
num_segments = size(sorted_waypoints, 1); % 起点 -> 第1个必经点 -> ... -> 终点
all_x = x0; % 用于保存所有路径点的x坐标
all_y = y0; % 用于保存所有路径点的y坐标

for i = 1:num_segments-1
   
   
        % 中间段: 必经点之间
        x_start = sorted_waypoints(i , 1);
        y_start = sorted_waypoints(i , 2);
        x_end = sorted_waypoints(i+1, 1);
        y_end = sorted_waypoints(i+1, 2);
    
    % 检查两点之间是否有障碍物
    if ~check_obstacle(x_start, y_start, x_end, y_end,100,param.obstacles)
        % 如果没有障碍物，直接连接这两个点
        x_path = [x_start; x_end];
        y_path = [y_start; y_end];
    else
        % 如果有障碍物，调用路径规划函数
                 p_start=[x_start;y_start];
                 p_end=[x_end; y_end];
                result = RRTstar12(param, p_start, p_end);  % 正确的调用
                 x_path = transpose(result.refinedP(1,:));%转置
              y_path = transpose(result.refinedP(2,:));
              
               
            
    end

    
    % 累积路径点
    all_x = [all_x; x_path(2:end)];
    all_y = [all_y; y_path(2:end)];
    
    % 计算路径长度
    path_length = 0;
    for j = 1:length(x_path) - 1
        path_length = path_length + norm([x_path(j+1), y_path(j+1)] - [x_path(j), y_path(j)]);
    end
    total_path_length = total_path_length + path_length;
    
end
% 显示总路径长度和总时间
disp(['总路径长度: ', num2str(total_path_length)]);


% 绘制路径
figure;
% plot(all_x, all_y, '--bo', 'LineWidth', 2);
plot(all_x, all_y, '--', 'LineWidth', 2);
hold on; 

% 标记起点、必经点和终点
plot(x0, y0, 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g'); % 起点
plot(sorted_waypoints(:, 1), sorted_waypoints(:, 2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'y'); % 必经点
plot(xf, yf, 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r'); % 终点

% 绘制障碍物
theta = linspace(0, 2*pi, 100);
for i = 1:num_obs
    xp = c(i) + a(i) * cos(theta);
    yp = d(i) + b(i) * sin(theta);
    plot(xp, yp, '.r');
end

% 设置图形参数
xlabel('x');
ylabel('y');
title('无人机路径规划');
axis equal;
grid on;

% 在图上添加文本
text(x0 + 0.2, y0, '起点', 'Color', 'black', 'HorizontalAlignment', 'left');
text(xf + 0.2, yf+0.5, '终点', 'Color', 'black', 'HorizontalAlignment', 'right');
for i = 2:size(sorted_waypoints, 1)-1
    text(sorted_waypoints(i, 1) + 0.2, sorted_waypoints(i, 2), ['Waypoint', num2str(i-1)], 'Color', 'black', 'HorizontalAlignment', 'left');
end
total_time=toc();
disp(['总时间: ', num2str(total_time), ' 秒']);
