% 检查两个点之间是否有障碍物
function has_obstacle = check_obstacle(x_start, y_start, x_end, y_end,n,obstacles)
    has_obstacle = false;
    for i = 1:length(obstacles(:,1))
        % 检查障碍物是否与起点和终点之间的直线相交
        if line_intersects_ellipse(x_start, y_start, x_end, y_end,n,obstacles)
            has_obstacle = true;
            break;
        end
    end
end

% 判断两点之间的直线是否与椭圆相交
function intersects = line_intersects_ellipse(x1, y1, x2, y2,n,obstacles)
   intersects = 0;
        dx = (x2-x1) / n; %计算每段线段的步长 dx 和 dy，n表示分多少段
        dy = (y2-y1) / n;
        for i = 1:n
            if isObstacleFree([x1+i*dx, y1+i*dy],obstacles) %n段中，第i段的坐标，判断其是否触碰障碍
               intersects = 1;
                break
            end
        end
end
 