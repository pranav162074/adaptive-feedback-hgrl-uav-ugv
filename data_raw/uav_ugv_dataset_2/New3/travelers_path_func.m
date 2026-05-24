function traveler_path=travelers_path_func(path, dist_matrix, n_travelers,near_zone,traveleridx,t)
% 为每个旅行商分配路径
        if t==n_travelers
             traveler_path=path(traveleridx(t):end);
             traveler_path_length=length(traveler_path);
              i=1;
            while traveler_path_length%删除其他无人机的任务
                    for j=1:n_travelers-1
                        if dist_matrix(path(traveleridx(j)),traveler_path(i))<near_zone
                            traveler_path(i)=[];
                            i=i-1;
                            break;
                        end
                    end
                     i=i+1;
                    traveler_path_length=traveler_path_length-1;
            end
           other_path=path(1:traveleridx(t)-1);%添加近距离点
                for i=1:length(other_path)
                    if dist_matrix(path(traveleridx(t)),other_path(i))<near_zone
                        traveler_path=[traveler_path,other_path(i)];
                    end
                end
                 traveler_path = [ traveler_path ,path(traveleridx(t))];
            
        elseif t==1
            traveler_path = path(traveleridx(t):(traveleridx(t+1)-1));
             traveler_path_length=length(traveler_path);
              i=1;
            while traveler_path_length%删除其他无人机的任务
                    for j=2:n_travelers
                        if dist_matrix(path(traveleridx(j)),traveler_path(i))<near_zone
                            traveler_path(i)=[];
                            i=i-1;
                            break;
                        end
                    end
                     i=i+1;
                    traveler_path_length=traveler_path_length-1;
            end
           other_path=path(traveleridx(t+1):end);
                for i=1:length(other_path)
                    if dist_matrix(path(traveleridx(t)),other_path(i))<near_zone
                        traveler_path=[traveler_path,other_path(i)];
                    end
                end
                 traveler_path = [ traveler_path ,path(traveleridx(t))];
            
        else
            
                traveler_path = path(traveleridx(t):(traveleridx(t+1)-1));%
                traveler_path_length=length(traveler_path);
                 i=1;
               while traveler_path_length
                    for j=1:t-1
                        if dist_matrix(path(traveleridx(j)),traveler_path(i))<near_zone
                            traveler_path(i)=[];
                            i=i-1;
                            break;
                        end
                    end
                        i=i+1;
                    traveler_path_length=traveler_path_length-1;
               end
                traveler_path_length=length(traveler_path);
                i=1;
               while traveler_path_length
                    for j=t+1:n_travelers
                        if dist_matrix(path(traveleridx(j)),traveler_path(i))<near_zone
                            traveler_path(i)=[];
                            i=i-1;
                            break;
                        end
                    end
                        i=i+1;
                    traveler_path_length=traveler_path_length-1;
               end
                other_path=[path(1:traveleridx(t)-1),path(traveleridx(t+1):end)];
                for i=1:length(other_path)
                    if dist_matrix(path(traveleridx(t)),other_path(i))<near_zone
                        traveler_path=[traveler_path,other_path(i)];
                    end
                end
                 traveler_path = [ traveler_path ,path(traveleridx(t))];
               
        end
end