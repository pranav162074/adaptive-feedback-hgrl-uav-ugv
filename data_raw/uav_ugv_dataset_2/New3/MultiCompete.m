clear; close all;
% 初始化参数

load('multiC3.mat','xMatrix1','yMatrix1','best_path1','startpoints','waypoints','param');
load('GAmultiC3.mat','xMatrix','yMatrix','best_path','oudist_matrix','near_zone');

m=size(param.obstacles,1);%障碍数
n_travelers = size(startpoints,1); % 旅行商数量
alpha=0.5;
beta=0.5;
for i=1:m
a(i)= param.obstacles(i,1); % 长轴
b(i) = param.obstacles(i,2); % 短轴
c (i)= param.obstacles(i,3); % 椭圆中心x坐标
d (i)= param.obstacles(i,4); % 椭圆中心y坐标
end
points = [startpoints;waypoints];
n = size(points, 1); % 总点数（包括起点和终点）


figure

title('case3');
hold on;

plot(waypoints(:,1),waypoints(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g'); % 经停点
hold on;
plot(startpoints(:,1),startpoints(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r'); % 起点
theta = linspace(0, 2*pi, 200);
for i = 1:m
    xp = c(i) + a(i) * cos(theta);
    yp = d(i) + b(i) * sin(theta);
    plot(xp, yp, '.r');
end


for t=1:n_travelers
    traveleridx=find(best_path<=n_travelers,n_travelers);
    traveler_path=travelers_path_func(best_path, oudist_matrix, n_travelers,near_zone,traveleridx,t);
    current_points=points(traveler_path,:);
    num_points=size(current_points,1);
     all_x = current_points(1,1); % 用于保存所有路径点的x坐标
all_y = current_points(1,2); % 用于保存所有路径点的y坐标
    for i=1:num_points-1
        x_start = current_points(i, 1);
         y_start = current_points(i, 2);
         x_end = current_points(i+1, 1);
         y_end = current_points(i+1, 2);
             x_path=[];%清空上一次的路径
             y_path=[];
       for j=1:size(xMatrix,3)
                 if xMatrix(traveler_path(i),traveler_path(i+1),j)==0 && yMatrix(traveler_path(i),traveler_path(i+1),j)==0&&j~=1
                     break;
                 else
              x_path(j)=xMatrix(traveler_path(i),traveler_path(i+1),j);
               y_path(j)=yMatrix(traveler_path(i),traveler_path(i+1),j);
                 end
       end
       if current_points(1,1)==0&&current_points(1,2)==0&&i==num_points-1%路径回到起点（0，0）
           x_path(j+1)=0;
           y_path(j+1)=0;
       end
       all_x = [all_x, x_path(2:end)];
    all_y = [all_y, y_path(2:end)];  
    end
    plot(all_x,  all_y, '-', 'LineWidth', 2,'Color','green'); 
    hold on;
    
            for i = 2:(length(traveler_path) - 1)
          text(points(traveler_path(i), 1) + 0.3, points(traveler_path(i), 2), ['P', num2str(i-1)], 'Color', 'black', 'HorizontalAlignment', 'left');
            end
end



for t=1:n_travelers
    traveleridx=find(best_path1<=n_travelers,n_travelers);
            if t==n_travelers
             traveler_path=[best_path1(traveleridx(t):end),best_path1(traveleridx(t))];
            else
                traveler_path = [best_path1(traveleridx(t):(traveleridx(t+1)-1)),best_path1(traveleridx(t))];
            end
   
    current_points=points(traveler_path,:);
    num_points=size(current_points,1);
     all_x = current_points(1,1); % 用于保存所有路径点的x坐标
all_y = current_points(1,2); % 用于保存所有路径点的y坐标
    for i=1:num_points-1
        x_start = current_points(i, 1);
         y_start = current_points(i, 2);
         x_end = current_points(i+1, 1);
         y_end = current_points(i+1, 2);
             x_path=[];%清空上一次的路径
             y_path=[];
       for j=1:size(xMatrix1,3)
                 if xMatrix1(traveler_path(i),traveler_path(i+1),j)==0 && yMatrix1(traveler_path(i),traveler_path(i+1),j)==0&&j~=1
                     break;
                 else
              x_path(j)=xMatrix1(traveler_path(i),traveler_path(i+1),j);
               y_path(j)=yMatrix1(traveler_path(i),traveler_path(i+1),j);
                 end
       end
       if current_points(1,1)==0&&current_points(1,2)==0&&i==num_points-1%路径回到起点（0，0）
           x_path(j+1)=0;
           y_path(j+1)=0;
       end
       all_x = [all_x, x_path(2:end)];
    all_y = [all_y, y_path(2:end)];  
    end
    plot(all_x,  all_y, '--', 'LineWidth', 2,'Color',[0.5,0.2,0.8]); 
    hold on;
          for i = 2:(length(traveler_path) - 1)
          text(points(traveler_path(i), 1) , points(traveler_path(i), 2)+0.5, ['W', num2str(i-1)], 'Color', 'black', 'HorizontalAlignment', 'left');
          end
    
end






