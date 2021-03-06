clc;
clear all;
close all;
%visual odometry initialization
%initializing parameters
addpath(".\plot")

%% Setup
ds = 1; % 0: KITTI, 1: Malaga, 2: parking

kitti_path = '../datasets/kitti';
malaga_path = '../datasets/malaga-urban-dataset-extract-07';
parking_path = '../datasets/parking';
addpath(genpath(pwd));

if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    assert(exist(kitti_path, 'dir') ~= 0);
    ground_truth = load([kitti_path '/poses/05.txt']);
    figure(10)
    ground_truth = plot(ground_truth(:,end-8),ground_truth(:,end));
    axis equal
    last_frame = 2760;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
    thr =2.1;
    maxReproj =8;
elseif ds == 1
    % Path containing the many files of Malaga 7.
    assert(exist(malaga_path, 'dir') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
    thr =1.5;
    maxReproj =5;
elseif ds == 2
    % Path containing images, depths and all...
    assert(exist(parking_path, 'dir') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
    thr= 2.1; 
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    maxReproj =1;
else
    assert(false);
end

%% Bootstrap
% need to set bootstrap_frames
bootstrap_frames(1) = 1;
bootstrap_frames(2) = 7;
cameraParams = cameraParameters('IntrinsicMatrix',K');
if ds == 0
    img0 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    P_R = detectSIFTFeatures(img0);
    P_R = P_R.Location;
    
    %initialization of the tracker 
    trackerP = vision.PointTracker('MaxBidirectionalError',1);
    initialize(trackerP,P_R,img0);
    
    % %tracking points
    %track  intermediate frames
    for i = bootstrap_frames(2)+1:bootstrap_frames(2)-1
        %read a neww frame
        I = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',i)]);
    
        %track point from old frame to new frame
        [~,~] = trackerP(I);
    end
    %track final frame
    I = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    P_R = detectSIFTFeatures(img0);
    P_R = P_R.Location;
    
    %initialization of the tracker 
    trackerP = vision.PointTracker('MaxBidirectionalError',1);
    initialize(trackerP,P_R,img0);
    
    % %tracking points
    %track  intermediate frames
    for i = bootstrap_frames(2)+1:bootstrap_frames(2)-1
        %read a neww frame
        I = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(i).name]));
    
        %track point from old frame to new frame
        [~,~] = trackerP(I);
    end
    %track final frame
    I = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
    P_R = detectSIFTFeatures(img0);
    P_R = P_R.Location;
    
    %initialization of the tracker 
    trackerP = vision.PointTracker('MaxBidirectionalError',1);
    initialize(trackerP,P_R,img0);
    
    % %tracking points
    %track  intermediate frames
    for i = bootstrap_frames(2)+1:bootstrap_frames(2)-1
        %read a neww frame
        I = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',i)]));
    
        %track point from old frame to new frame
        [~,~] = trackerP(I);
    end
    %track final frame
    I = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(2))]));
else
    assert(false);
end

[P_F,keepP] = trackerP(I);
P_F = P_F(keepP,:);
P_R = P_R(keepP,:);

%essential matrix estimation 
[E,inliersIndex] = estimateEssentialMatrix(P_R,P_F,cameraParams);
P_F = P_F(inliersIndex,:);
P_R = P_R(inliersIndex,:);

%Camera pose extraction
[R_C_W,t_W_C] = relativeCameraPose(E,cameraParams,P_R,P_F);

%triangulate 3d points
[P_i_1,X_i_1,C_i_1,f_i_1,T_i_1] = triangulate2dto3d(P_F,P_R,repmat([reshape(eye(3),1,[]),0,0,0],size(P_R,1),1),R_C_W, t_W_C,K,thr);

%plot 3d points and camera poses
figure(3)
plot3(X_i_1(:,1),X_i_1(:,2),X_i_1(:,3),"mo");
plotCoordinateFrame(R_C_W', t_W_C', 2);
plotCoordinateFrame(eye(3), [0 0 0]', 2);

%plot points associated to the 3d points
figure (4)
imshow(I)
hold on
plot(P_i_1(:,1),P_i_1(:,2),"gx");
hold off

%% Continuous operation

%initialize tracker for P
trackerP = vision.PointTracker('MaxBidirectionalError',1);
initialize(trackerP,P_i_1,I);

%initialize tracker for C
trackerC = vision.PointTracker('MaxBidirectionalError',1);
%initialize(trackerC,C_i_1,I);

%save history of poses and orientation
histloc = zeros(last_frame-bootstrap_frames(2),3);

range = (bootstrap_frames(2)+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        I = imread([kitti_path '/05/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        I = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        I = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end  
        %track point from old frame to new frame
    [P_i,keepP] = trackerP(I);
    P_i = P_i(keepP,:);
    P_i_1 = P_i_1(keepP,:);
    X_i = X_i_1(keepP,:);

    %estimate pose
    [R_C_W,t_W_C,keepP] = estimateWorldCameraPose(P_i,X_i,cameraParams,"MaxReprojectionError",maxReproj);
    P_i = P_i(keepP,:);
    P_i_1 = P_i_1(keepP,:);
    X_i = X_i(keepP,:);
    %save pose
    %histori (end+1,:,:)=R_C_W';
    histloc (i-bootstrap_frames(2),:)=t_W_C;

    %plot pose and 3d points
    figure(9) 
    plot(X_i(:,1),X_i(:,3),"mo");
    hold on
    plot(histloc(1:i-bootstrap_frames(2),1),histloc(1:i-bootstrap_frames(2),3),"r-")
    hold off
    axis equal


    newP=[];
    if i ~=bootstrap_frames(2)+1

        %track point from old frame to new frame
        [C_i,keepC] = trackerC(I);
        C_i = C_i(keepC,:);
        f_i = f_i_1(keepC,:);
        T_i = T_i_1(keepC,:);

        %triangulate new points
        [newP,newX,C_i,f_i,T_i]=triangulate2dto3d(C_i,f_i,T_i,R_C_W, t_W_C,K,thr);
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

    release(trackerP);
    release(trackerC);

    initialize(trackerP,P_i_1,I);
    initialize(trackerC,C_i_1,I);
    pause(0.01);
  
end