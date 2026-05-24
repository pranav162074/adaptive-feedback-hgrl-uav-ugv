clear; close all;

% 初始条件
x0 = 0; y0 = 0; % 起点
xf = 10; yf = 10; % 终点
waypoints = [ 4,5 ;9,6; 9,10 ;6,6; 5,9]; % 经停点
pri_points=[8,5];%优先点
k=size(pri_points,1);
H = 10; % 每段的时间步长
V = 1; % 无人机速度
u_max = 0.2; % 最大加速度
stop_time = 0; % 每个必经点的停留时间

param.searchFeild = [-3,12,-3,12];   % 搜索区域
param.threshold = 1;   % d_{min}
param.theta = 5*pi/6;    % 论文中theta取150度
param.resolution = 10;   % 决定在路径上多远取几个点，以判断路径是否避开障碍
param.maxNodes = 5000;  %算法在停止之前会生成的最大节点数，最大迭代次数
param.step_size = 1;  %新节点向采样点生长时的最大步长
param.neighbourhood = 1; % 设置neighbourhood的半径r
param.random_seed =40;
param.u_max = 0.2;    %最大曲率  
param.obstacles=[2,1,2,3;
  3,3,7,2;
  2,2,2,7;
  2,2,8,8;];
% 障碍参数（椭圆）
a = [2, 3, 2,2 ]; % 长轴
b = [1, 3, 2, 2]; % 短轴
c = [2, 7, 2, 8]; % 椭圆中心x坐标
d = [3, 2, 7, 8]; % 椭圆中心y坐标
m = size(a, 2); % 障碍物数量

% 使用优化必经点顺序
% [sorted_waypoints, total_distance] = Gene2(x0, y0, xf, yf, waypoints,param);
[sorted_waypoints, total_distance] = Gene2(pri_points(k,1),pri_points(k,2), xf, yf, waypoints,param);
sorted_waypoints=[x0,y0;pri_points(1:k-1,:);sorted_waypoints];
% 初始化路径长度和时间
total_path_length = 0;
total_time = 0;

% 路径规划处理多个路径段
num_segments = size(sorted_waypoints, 1); 
all_x = x0; % 用于保存所有路径点的x坐标
all_y = y0; % 用于保存所有路径点的y坐标

for i = 1:num_segments-1
     x_start = sorted_waypoints(i, 1);
         y_start = sorted_waypoints(i, 2);
         x_end = sorted_waypoints(i+1, 1);
         y_end = sorted_waypoints(i+1, 2);
    % 检查两点之间是否有障碍物
    if ~check_obstacle(x_start, y_start, x_end, y_end,100,param.obstacles)
        % 如果没有障碍物，直接连接这两个点
        x_path = [x_start; x_end];
        y_path = [y_start; y_end];
    else
        % 如果有障碍物，调用路径规划函数
        output = solve_problem(x_start, y_start, x_end, y_end, H, a, b, c, d, V, u_max);
         % 累积时间
   total_time = total_time + output.time+ stop_time;
        % 提取路径点
        x_path = [x_start; output.x(1:H-2); x_end];
        y_path = [y_start; output.x(H-1:2*H-4); y_end];
        
        %判断irm结果是否符合要求
        for j=1:H-2
            if ~isObstacleFree ([output.x(j),output.x(H-2+j)],param.obstacles)
                 x_path = [x_start; output.x(1:H-2); x_end];
                 y_path = [y_start; output.x(H-1:2*H-4); y_end];
                 irm_free=1;
            else
                 p_start=[x_start;y_start];
                 p_end=[x_end; y_end];
                result = RRTstar12(param, p_start, p_end);  % 正确的调用
                 x_path = transpose(result.refinedP(1,:));%转置
              y_path = transpose(result.refinedP(2,:));
              total_time = total_time + result.time_rrtstar;
                break
            end
        end       
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
disp(['总时间: ', num2str(total_time), ' 秒']);

% 绘制路径
figure;
plot(all_x, all_y, '--bo', 'LineWidth', 2);
hold on;

% 标记起点、必经点和终点
plot(x0, y0, 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g'); % 起点
plot(sorted_waypoints(:, 1), sorted_waypoints(:, 2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'y'); % 必经点
plot(xf, yf, 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r'); % 终点

% 绘制障碍物
theta = linspace(0, 2*pi, 100);
for i = 1:m
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
    text(sorted_waypoints(i, 1) + 0.2, sorted_waypoints(i, 2), ['必经点', num2str(i-1)], 'Color', 'black', 'HorizontalAlignment', 'left');
end
