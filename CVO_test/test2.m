%skip this part served for tests
%----------------------------------------------------------------------------
% % clear all;
% % clc;
% % close all;
% % addpath("C:\Users\samso\OneDrive\Bureau\etude\VAMR\Exercise 7 - From images to localization\code\plot")
% % initialize the camera
% IntrinsicMatrix = [ 7.188560000000e+02 0 6.071928000000e+02
%                     0 7.188560000000e+02 1.852157000000e+02
%                     0 0 1]';
% cameraParams = cameraParameters('IntrinsicMatrix',IntrinsicMatrix);
% 
% % read first image
% I_R = imread('C:\Users\samso\OneDrive\Bureau\etude\VAMR\Exercise 7 - From images to localization\data\000000.png');
% 
% %extract the 100 best feature
% % P_i_1 = detectHarrisFeatures(I_R);
% % P_i_1 = selectStrongest(P_i_1,100);
% % P_i_1 = P_i_1.Location;
% P_i_1 = fliplr(load('C:\Users\samso\OneDrive\Bureau\etude\VAMR\Exercise 7 - From images to localization\data\keypoints.txt'));
% X_i_1 = load('C:\Users\samso\OneDrive\Bureau\etude\VAMR\Exercise 7 - From images to localization\data\p_W_landmarks.txt');
% row = find(P_i_1(:,1)>500 & P_i_1(:,1)<630);
% P_i_1(row,:)= [] ;
% X_i_1(row,:)= [] ;
% % P_i_1 = P_i_1(1:200,:);
% % X_i_1 = X_i_1(1:200,:);
% C_i_1 = [];
% f_i_1 = [];
% T_i_1 = [];

% %plot features
% figure(8);
% imshow(I_R);
% hold on;
% plot(P_i_1(:,1),P_i_1(:,2),"mo");
% hold off;
% pause(0.1);
% figure(9)
% plot3(X_i_1(:,1),X_i_1(:,2),X_i_1(:,3),"mo");


%-------------------------------------------------------------------------
%resume from here
%initialize the visual odometry
voinit

%initialize tracker for P
trackerP = vision.PointTracker('MaxBidirectionalError',1);
initialize(trackerP,P_i_1,I);

%initialize tracker for C
trackerC = vision.PointTracker('MaxBidirectionalError',1);
%initialize(trackerC,C_i_1,I);

%save history of poses and orientation
histori = [];
histloc = [];

for i = idx+1:2760
    %read a new frame
    I = imread(sprintf('%s%06d.png',str,i));

    %track point from old frame to new frame
    [P_i,keepP] = trackerP(I);
    P_i = P_i(keepP,:);
    P_i_1 = P_i_1(keepP,:);
    X_i = X_i_1(keepP,:);

    %estimate pose
    [R_C_W,t_W_C,keepP] = estimateWorldCameraPose(P_i,X_i,cameraParams);
    P_i = P_i(keepP,:);
    P_i_1 = P_i_1(keepP,:);
    X_i = X_i(keepP,:);

    %save pose
    histori (end+1,:,:)=R_C_W';
    histloc (end+1,:)=t_W_C;

    %plot pose and 3d points
    figure(9) 
    plot3(X_i(:,1),X_i(:,2),X_i(:,3),"mo");
    hold on
    for j = 1:size(histloc,1)
        %plotCoordinateFrame(reshape(histori(j,:,:),3,3), histloc(j,:)', 2);
       plot3(histloc(:,1),histloc(:,2),histloc(:,3),"r-")
    end
    hold off
   % xlim([0,50])
   % zlim([-5,20])
    view(0,0)

    newP=[];
    if i ~=idx+1

        %track point from old frame to new frame
        [C_i,keepC] = trackerC(I);
        C_i = C_i(keepC,:);
        C_i_1 = C_i_1(keepC,:);
        f_i = f_i_1(keepC,:);
        T_i = T_i_1(keepC,:);

        %triangulate new points
        [newP,newX,C_i,f_i,T_i]=triangulate2(C_i,f_i,T_i,R_C_W, t_W_C,IntrinsicMatrix',C_i_1,R_C_W_1, t_W_C_1);
        P_i=[P_i;newP];
        X_i=[X_i;newX];
    else
        C_i = [];
        f_i =[];
        T_i =[];
    end
    
    
    %extract new feature
    corners = detectSIFTFeatures(I);
    %corners =corners.selectStrongest(200);
    corners = corners.Location;
    %row = find(corners(:,1)>500 & corners(:,1)<630);
    %corners(row,:)= [] ;

    %Verifing that the features aren't already tracked
    for j = 1:size(P_i,1)
       row = find (corners(:,1)<=P_i(j,1)+8 & corners(:,1)>=P_i(j,1)-8 & corners(:,2)<=P_i(j,2)+8 & corners(:,2)>=P_i(j,2)-8);
       corners (row,:) = [];
    end
    for j = 1:size(C_i,1)
        row = find (corners(:,1)<=C_i(j,1)+8 & corners(:,1)>=C_i(j,1)-8 & corners(:,2)<=C_i(j,2)+8 & corners(:,2)>=C_i(j,2)-8);
        corners (row,:) = [];
    end
    
    %updating C_i,f_i,T_i
    C_i = [C_i;corners];
    f_i = [f_i;corners];
    t = [reshape(R_C_W',1,[]),t_W_C];
    T_i = [T_i;repmat(t,size(corners,1),1)];
    %display P and C and old matching points
    figure (8) 
    imshow(I);
    hold on
    plotMatches(1:size(P_i_1,1), flipud(P_i(1:size(P_i_1,1),:)'), flipud(P_i_1'))
%     plotMatches(1:size(f_i,1), flipud(C_i'), flipud(f_i'))
%     plot(C_i(:,1),C_i(:,2),"mo");
    plot(P_i(:,1),P_i(:,2),"gx");
    if numel(newP)>0
    plot(newP(:,1),newP(:,2),"bs");
    end
    hold off

    %reseting variables
    P_i_1 = P_i;
    X_i_1 = X_i;
    C_i_1 = C_i;
    f_i_1 = f_i;
    T_i_1 = T_i;
    R_C_W_1 =R_C_W;
    t_W_C_1 =t_W_C;

    release(trackerP);
    release(trackerC);

    initialize(trackerP,P_i_1,I);
    initialize(trackerC,C_i_1,I);
    pause(0.1);

end