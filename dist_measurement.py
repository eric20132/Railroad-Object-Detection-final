import cv2
import numpy as np
import os
import math
import scipy
import random
import matplotlib.pyplot as plt
import geopy.distance
from serial import Serial
import pynmea2
import torch
import zmq

def find_displacement(coord_1,coord_2):
    #distance calculation
    dis = geopy.distance.geodesic(coord_1,coord_2).m
    return dis


def find_latlong(line,start):
    
    if start:
        count = 0
        #print('Establishing RTK GPS.....')
        while True:
            count += 1
            if count == 20:
                return
    while True:
        #print(type(line))
        #print(line.startswith('$GNGGA'))
        if line.startswith('$GNGGA'):
            #print(line)
            try:
                dollar_sign = line.find('$')
                #print(dollar_sign)
                lat_start = dollar_sign + 7 + 10
                lat_end = lat_start + 10
                #print(lat_start,lat_end)
                latitude = float(line[lat_start:lat_end])/100
                #print(line[lat_start:lat_end])
                
                long_start = lat_end + 3
                long_end = long_start + 10
                #print(long_start,long_end)
                longitute = float(line[long_start:long_end+1])/100
                #print(line[long_start:long_end+1])
                return latitude,longitute
            except:
                return (0,0)
        #print (latitude,longitude)
        


def shift_points(frame1,frame2):
    img_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img_gray,None)
    img_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints2, descriptors2 = sift.detectAndCompute(img_gray,None)
    return keypoints1,descriptors1,keypoints2, descriptors2


def matching_points(keypoints1,descriptors1,keypoints2,descriptors2,img1,img2,output_file_name,op_needed = False):
    bruteforce = cv2.BFMatcher( cv2.NORM_L2, crossCheck=True)
    matching_points = bruteforce.match(descriptors1,descriptors2)
    def sort_tuples(matching_points):
        matching_points = list(matching_points)
        for i,point1 in enumerate(matching_points):
            for j,point2 in enumerate(matching_points):
                if point1.distance<point2.distance and j < i:
                    matching_points[i],matching_points[j] = matching_points[j],matching_points[i]
        return tuple(matching_points)
    matching_points = sort_tuples(matching_points)[:500]
    #img_out = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matching_points,img2)
      
    #cv2.imwrite(output_file_name,img_out)
    image_coordinates_matched = []
    for i in range(len(matching_points)):
        x,y  = keypoints1[matching_points[i].queryIdx].pt
        x_1,y_1 = keypoints2[ matching_points[i].trainIdx ].pt 
        image_coordinates_matched.append([x,y,x_1,y_1])
    return image_coordinates_matched



#Calculating homography using Linear Least Squares method
def calculate_homography(domain_coords,range_coords):
    #Number of points - variable to store 
    number_of_points = len(domain_coords)
    
    #Equation Ax=b can be solved by using x = inv(A)b
    
    #Creation of A and b matrices from the coordinate points set
    b = np.zeros(number_of_points*2)
    A = np.zeros((number_of_points*2,8))
    p = 0
    #Creation of A matrix
    for i,point in enumerate(domain_coords):
        A[p][0] = point[0]
        A[p][1] = point[1]
        A[p][2] = 0
        A[p][3] = 0
        A[p][4] = 1
        A[p][5] = 0
        A[p][6] = -point[0]*range_coords[i][0]
        A[p][7] = -point[1]*range_coords[i][0]
        A[p+1][0] = 0
        A[p+1][1] = 0
        A[p+1][2] = point[0]
        A[p+1][3] = point[1]
        A[p+1][4] = 0
        A[p+1][5] = 1
        A[p+1][6] = -point[0]*range_coords[i][1]
        A[p+1][7] = -point[1]*range_coords[i][1]
        p += 2
    
    #Creation of b vector
    p = 0
    for i,point in enumerate(range_coords):
        
        b[p] = point[0]
        b[p+1] = point[1]
        p += 2
    #a_inv = np.linalg.pinv(A)
    #a = np.matmul(a_inv,b)
    
    #Solving AX=b to get X
    x = np.linalg.pinv(A)
    x = np.matmul(x,b)
    #Creating Homography matrix using x
    H = np.array(([x[0],x[1],x[4]],[x[2],x[3],x[5]],[x[6],x[7],1]))
    return (H)

def check_no_of_outliers(homography,point_sets,delta):
    inliers_list = []
    outliers_list = []
    for point in point_sets:
        point_domain = np.array([point[0],point[1],1])
        point_predicted = np.matmul(homography,point_domain)
        point_predicted = point_predicted/point_predicted[2]
        point_range = np.array([point[2],point[3],1])
        ##print('point_predicted',point_predicted)
        error = np.linalg.norm((point_range - point_predicted))
        ##print('original',point_range)
        error = np.linalg.norm((point_range - point_predicted))
        if error < delta:
            inliers_list.append([point[0],point[1],point[2],point[3]])
        else:
             outliers_list.append([point[0],point[1],point[2],point[3]])
    return inliers_list,outliers_list
        

def calculate_homography_linear(point_sets,n,epsilon):
    #Calculating the minimum number of trials needed
    N = math.ceil(math.log(1-0.99)/math.log(1-math.pow((1-epsilon),n)))

    all_inliers_set = []
    set_of_nums = [i for i in range(len(point_sets))]
    
    # Randomly select n points 
    for i in range(N):
        random_set = random.choices(set_of_nums,k=n)
        domain_coords = []
        range_coords = []
        for index_number in random_set:
            domain_coords.append([point_sets[index_number][0],point_sets[index_number][1]])
            range_coords.append([point_sets[index_number][2],point_sets[index_number][3]])
        
        
        #calculating the homography - with ransac points
        homography_ransac_steps = calculate_homography(domain_coords,range_coords )
        list_of_inliers,list_of_outliers = check_no_of_outliers(homography_ransac_steps, point_sets, 4)
        #print(len(list_of_inliers),len(point_sets))
        
        if len(list_of_inliers) >= 0.5*len(point_sets):
            break
        if i == 100:
            break
    domain_coords_inliers = []
    range_coords_inliers = []

    #Calculating the complete homography using all points in the inlier set
    for point in list_of_inliers:
        domain_coords_inliers.append([point[0],point[1],1])
        range_coords_inliers.append([point[1],point[2],1])
    homography_linear = calculate_homography(domain_coords_inliers,range_coords_inliers)
    
    return homography_linear,list_of_inliers,list_of_outliers

# Function to plot the inliers and outliers 
def plot_inliers_outliers(img1, img2, inliers_list, outliers_list):
    h,w,z = img1.shape
    h1,w1,z = img2.shape
    img_op_inliers = np.concatenate((img1,img2),axis=1)
    for point in inliers_list:
        p1 = [int(point[0]),int(point[1])]
        p2 = [int(point[2]+w1),int(point[3])]
        cv2.circle(img_op_inliers,p1,3,(255,255,255),1)
        cv2.circle(img_op_inliers,p2,3,(255,255,255),1)
        cv2.line(img_op_inliers,p1,p2,(0,0,0),1)
    cv2.imwrite('inliers.jpg',img_op_inliers)
    img_op_outliers = np.concatenate((img1,img2),axis=1)
    for point in outliers_list:
        p1 = [int(point[0]),int(point[1])]
        p2 = [int(point[2]+w1),int(point[3])]
        cv2.circle(img_op_outliers,p1,3,(255,255,255),1)
        cv2.circle(img_op_outliers,p2,3,(255,255,255),1)
        cv2.line(img_op_outliers,p1,p2,(0,0,0),1)
    cv2.imwrite('outliers.jpg',img_op_outliers)

def cost_function(homography,list_of_inliers):
    domain_points = []
    range_points = []
    calculated_range_points = []
    homography = homography.reshape(3,3)
    for point in list_of_inliers:
        domain_points.append([point[0],point[1]])
        x = [point[0],point[1],1]
        y = np.matmul(homography,x)
        y = y/y[2]
        calculated_range_points.append(y[1])
        calculated_range_points.append(y[0])
        range_points.append(point[2])
        range_points.append(point[3])            
            
    error = np.array(range_points)-np.array(calculated_range_points)
    homography = homography.reshape((1,9))[0]
    ##print(range_points,calculated_range_points)
    
    return error
def transform_images(img_filename1,homography,output_image):
    img1 = cv2.imread(img_filename1)
    h_1, w_1, z = img1.shape
    x_max = 0
    x_min = 10000
    y_max = 0
    y_min = 10000
    t = np.linspace(0, w_1,w_1,endpoint=False)
    t1 = np.linspace(0, h_1,h_1,endpoint=False)
    pic1_points = []
    for i in t:
        for j in t1:
            x_arr = np.array(([i,j,1]))
            pic1_points.append([i,j])
            result_point =  np.matmul(homography,x_arr)
            x_max  = max(x_max,result_point[0]/result_point[2])
            y_max  = max(y_max,result_point[1]/result_point[2])
            x_min = min(x_min,result_point[0]/result_point[2])
            y_min = min(y_min,result_point[1]/result_point[2])
    
    
    h_output = math.ceil(y_max -y_min+1)
    w_output = math.ceil(x_max -x_min+1)
    output_img = np.full((h_output,w_output,z),0)
    for i in t:
        for j in t1:
            ##print(img1[int(i),int(j),:])
            img1_coords = np.array([i,j,1])
            result_coords = np.matmul(homography,img1_coords)
            x_ = result_coords[0]/result_coords[2] 
            y_ =  result_coords[1]/result_coords[2]
            x_ = x_ - int(x_min)
            y_ = y_ - int(y_min)
            result_coords = result_coords/result_coords[2]
            output_img[int(y_),int(x_),:] = img1[int(j),int(i),:]
    cv2.imwrite(output_image,output_img)


#calculation of the A matrix to calculate F matrix
def calculate_A_matrix(pic1_points,pic2_points):
    A = []
    for i,point in enumerate(pic1_points):
        ##print(point)
        x = point[0]
        y = point[1]
        x_1 = pic2_points[i][0]
        y_1 = pic2_points[i][1]
        row = [x_1*x, x_1*y, x_1, y_1*x, y_1*y, y_1, x,y, 1 ]
        A.append(row)
    return A


def get_points(list_of_inliers):
    domain_coords_inliers = []
    range_coords_inliers = []
    for point in list_of_inliers:
        domain_coords_inliers.append([point[0],point[1]])
        range_coords_inliers.append([point[1],point[2]])
    return domain_coords_inliers,range_coords_inliers


def epipole(F):
    u,s,vh = np.linalg.svd(F)
    epipole_left = np.transpose(vh[-1,:])
    epipole_right = u[:,-1]
    epipole_left = epipole_left/epipole_left[2]
    epipole_right = epipole_right/epipole_right[2]
    return epipole_left, epipole_right

def generate_p_right(epipole,F):
    e_x = [[0, -epipole[2], epipole[1]],[epipole[2], 0, -epipole[0]],[-epipole[2], epipole[0], 0]]
    cross_output = np.matmul(e_x,F)
    ##print(cross_output)
    p_right = np.zeros((3,4))
    p_right[0,0:3] = cross_output[0,:]
    p_right[1,0:3] = cross_output[1,:]
    p_right[2,0:3] = cross_output[2,:]
    p_right[0,3] = epipole[0]
    p_right[1,3] = epipole[1]
    p_right[2,3] = epipole[2]
    return p_right

def calculate_world_points(pic1_points,pic2_points,P_l,P_r):
    world_coordinates = []
    ##print(P_l,P_l[0,:])
    for i,point_left in enumerate(pic1_points):
        x_l = point_left[0]
        y_l = point_left[1]
        x_r = pic2_points[i][0]
        y_r = pic2_points[i][1]
        A = np.zeros((4,4))
        A[0] = x_l*P_l[2,:] - P_l[0,:]
        A[1] = y_l*P_l[2,:] - P_l[1,:]
        A[2] = x_r*P_r[2,:] - P_r[0,:]
        A[3] = y_r*P_r[2,:] - P_r[1,:]
        ##print(A)
        A_TA = np.matmul(np.transpose(A),A)
        u,s,vh = np.linalg.svd(A_TA)
        world_coord = np.transpose(vh[-1,:])
        world_coord = [world_coord[0]/(world_coord[3]*1), world_coord[1]/(world_coord[3]*1), world_coord[2]/(world_coord[3]*1)]
        world_coordinates.append(world_coord)
    ##print(world_coordinates[0:10])
    return(world_coordinates)

def generate_3d_plot(real_world_coord):
    #print('real world coord',real_world_coord)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x_list = []
    y_list = []
    z_list = []
    for point in real_world_coord:
        x_list.append(point[0])
        y_list.append(point[1])
        z_list.append(point[2])
    ax.scatter3D(x_list,y_list,z_list)
    plt.show()
    plt.savefig('3D.jpg')


#Cost function to optimize P right using non linear optimization
def cost_function_1(P_right, P_left, F, pic1_points, pic2_points):
    P_right = np.reshape(P_right,(3,4))
    A = np.zeros((4,4))
    world_point_list = []
    for i,point1 in enumerate(pic1_points):
        point2 = pic2_points[i]
        x,y = point1[0],point1[1]
        x_,y_ = point2[0],point2[1]
        A[0] = x*P_left[2,:] - P_left[0,:]
        A[1] = y*P_left[2,:] - P_left[1,:]
        A[2] = x_*P_right[2,:] - P_right[0,:]
        A[3] = y_*P_right[2,:] - P_right[1,:]
        u,s,vh = np.linalg.svd(A)
        world_point =np.transpose(vh[-1,:])
        world_point_list.append(world_point)


    error_list = []
    for i,world_point in enumerate(world_point_list):
        reprojected_point = np.matmul(P_right,world_point)
        reprojected_point = reprojected_point/reprojected_point[2]
        x_err = (reprojected_point[0]-pic2_points[i][0])**2
        y_err = (reprojected_point[1]-pic2_points[i][1])**2
        error_list.append(x_err)
        error_list.append(y_err)
    return error_list

def transform_images_1(img,homography):
    in_arr = []
    out_arr = []
    h_1, w_1, z = img.shape
    x_max = 0
    x_min = 10000
    y_max = 0
    y_min = 10000
    t = np.linspace(0, w_1,w_1//4,endpoint=False)
    t1 = np.linspace(0, h_1,h_1//4,endpoint=False)
    for i in t:
        for j in t1:
            x_arr = np.array(([i,j,1]))
            in_arr.append([i,j])
            
            result_point =  np.matmul(homography,x_arr)
            x_max  = max(x_max,result_point[0]/result_point[2])
            y_max  = max(y_max,result_point[1]/result_point[2])
            x_min = min(x_min,result_point[0]/result_point[2])
            y_min = min(y_min,result_point[1]/result_point[2])
    
    
    h_output = math.ceil(y_max -y_min+1)
    w_output = math.ceil(x_max -x_min+1)
    for i in t:
        for j in t1:
            ##print(img1[int(i),int(j),:])
            img1_coords = np.array([i,j,1])
            result_coords = np.matmul(homography,img1_coords)
            x_ = result_coords[0]/result_coords[2] 
            y_ =  result_coords[1]/result_coords[2]
            x_ = x_ - int(x_min)
            y_ = y_ - int(y_min)
            result_coords = result_coords/result_coords[2]
            out_arr.append([x_,y_])
    return in_arr,out_arr

def points_needed(pic1_points,pic2_points,objects1,objects2):
    num_objs1 = objects1.shape[0]
    num_objs2 = objects2.shape[0]
    if num_objs1<num_objs2:
        num_objs = num_objs1
    else:
        num_objs = num_objs2
    point_complete = []
    point2_complete = []
    for obj_idx in range(num_objs):
        bounding_mat = objects1[obj_idx]
        #print(bounding_mat)
        pic_obj = []
        pic2_obj = []
        for i,pic1_point in enumerate(pic1_points):
            ##print(pic1_point[0])
            if pic1_point[0]>bounding_mat[0] and pic1_point[0]<bounding_mat[2] and pic1_point[1]>bounding_mat[1] and pic1_point[1]<bounding_mat[3]:
                pic_obj.append(pic1_point)
                pic2_obj.append(pic2_points[i])
        point_complete.append(pic_obj)
        point2_complete.append(pic2_obj)
    return point_complete,point2_complete

def min_max_vals(real_world_coords):
    x_list = []
    y_list = []
    z_list = []
    for pnt in real_world_coords:
        x_list.append(pnt[0])
        y_list.append(pnt[1])
        z_list.append(pnt[2])
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)
    x_min,x_max = np.min(x_list),np.max(x_list)
    y_min,y_max = np.min(y_list),np.max(y_list)
    z_min,z_max = np.min(z_list),np.max(z_list)
    return x_min,x_max,y_min,y_max,z_min,z_max

def create_img(img,objs,y_max_list,y_min_list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_ret = img.copy()
    for i,obj in enumerate(objs):
        if i == len(y_min_list):
            break
        img_ret = cv2.rectangle(img,(int(obj[0]),int(obj[1])),(int(obj[2]),int(obj[3])),(255,0,0),1)
        text = str(round(y_max_list[i]/100,2))+';'+str(round(y_min_list[i]/100,2))
        img_ret = cv2.putText(img_ret, text, (int(obj[0]),int(obj[1])), font,0.35, (0,255,0),1, cv2.LINE_AA)
    return img_ret



        


if __name__ == '__main__':
    context1 = zmq.Context()
    socket1 = context1.socket(zmq.SUB)
    socket1.connect('tcp://localhost:5556')
    socket1.setsockopt(zmq.SUBSCRIBE, str.encode(''))

    cap = cv2.VideoCapture(0)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    suc = True
    cnt = 0
    k_camera = np.array([[1257.08682,0,969.361185],[0,1255.84212,547.349395],[0,0,1]])
    dist_mat = np.array([[-3.18148781e-1, 8.27311268e-2, -3.07626401e-4, 1.40382016e-6, -2.20760346e-4]])
    stream = socket1.recv_string()
    find_latlong(stream,True)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    ret,img1 = cap.read()
    img1 = cv2.undistort(img1,k_camera,dist_mat,None,None)
    h,w,_ = img1.shape
    out = cv2.VideoWriter('out.avi',codec,30,(w,h))
    lat1, long1 = find_latlong(stream,False)
    while suc:
        print(cnt)
        print(lat1,long1)
        img_left = img1.copy()
        results1 = model(img_left)
        #results1.##print()
        

        objects1 = results1.xyxy[0]
        objects1 = objects1.cpu().numpy()
        ##print(objects1)

        stream = socket1.recv_string()
        lat2, long2 = find_latlong(stream,False)
        ret,img2 = cap.read()
        img2 = cv2.undistort(img2,k_camera,dist_mat,None,None)
        img_right = img2.copy()
        results2 = model(img_right)
        #results1.##print()
        

        objects2 = results2.xyxy[0]
        objects2 = objects2.cpu().numpy()
        # ##print(objects2)
        dist = find_displacement((lat1,long1),(lat2,long2))*1000
        #dist = 200
        print('dist',dist)
        keypoints1,descriptors1,keypoints2, descriptors2 = shift_points(img1,img2)
        img_corres = matching_points(keypoints1,descriptors1,keypoints2,descriptors2,img1,img2,'matched.jpg')
        homography_linear,list_of_inliers1,outliers = calculate_homography_linear(img_corres, n = 5, epsilon= 0.75)
        if len(list_of_inliers1) < 100:
            img1 = img2.copy()
            #lat1, long1 = lat2,long2
            continue
        #plot_inliers_outliers(img1,img2,list_of_inliers1,outliers)
        #homography_nonlinear = np.linalg.inv(homography_nonlinear)
        ##print('list_of_inliers1',list_of_inliers1)

        pic1_points,pic2_points = get_points(list_of_inliers1)
        

        

        #Calculate P_left and P_right + optimize
        P_left = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        P_left = np.matmul(k_camera,P_left)
        
        R_t = np.array([[1,0,0,dist],[0,1,0,0],[0,0,1,0]])
        P_right = np.matmul(k_camera,R_t)
        ##print('P left',P_left)
        ##print('P right', P_right)
        ##print(epipole_left,epipole_right_optimized)
        pic1_points,pic2_points = points_needed(pic1_points,pic2_points,objects1,objects2)
        #pic1_points,pic2_points = transform_images_1(img1, homography_nonlinear)
        ##print('pic1_points',pic1_points)
        ##print('pic2_points', pic2_points)
        num_objs = len(pic1_points)
        print('no of objs',num_objs)
        if num_objs == 0:
            continue
        y_min_list = []
        y_max_list = []
        flg = False
        for i in range(num_objs):
            obj_pic1 = pic1_points[i]
            obj_pic2 = pic2_points[i]
            
            world_coordinates = calculate_world_points(obj_pic1,obj_pic2,P_left, P_right)
            #print('world coords',world_coordinates)
            if len(world_coordinates) == 0:
                flg = True
                break
            x_min,x_max,y_min,y_max,z_min,z_max = min_max_vals(world_coordinates)
            y_min_list.append(y_min)
            y_max_list.append(y_max)
        print(lat1,long1,lat2,long2)
        #print(y_min_list)
        #print(y_max_list)
        img_opb = img2.copy()
        img_opb = create_img(img_opb,objects2,y_max_list,y_min_list)
        out.write(img_opb)
        #cv2.imshow('op',img_opb)
        #cv2.waitKey(1)
	
        img1 = img2.copy()
        lat1, long1 = lat2,long2
        cnt = cnt + 1
        
        if cnt == 150:
            
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


