import numpy as np
import sys
import os
import glob
import cv2
import operator
from pylab import *
from scipy.ndimage import filters
import scipy
from scipy import signal
import sys
from scipy.ndimage import interpolation

# read files
if len(sys.argv) > 1 and sys.argv[1] == "parrington":
    files = ["../parrington/prtn00.jpg", "../parrington/prtn01.jpg", "../parrington/prtn02.jpg", "../parrington/prtn03.jpg", "../parrington/prtn04.jpg", "../parrington/prtn05.jpg", "../parrington/prtn06.jpg", "../parrington/prtn07.jpg", "../parrington/prtn08.jpg", "../parrington/prtn09.jpg", "../parrington/prtn10.jpg", "../parrington/prtn11.jpg", "../parrington/prtn12.jpg", "../parrington/prtn13.jpg", "../parrington/prtn14.jpg", "../parrington/prtn15.jpg", "../parrington/prtn16.jpg", "../parrington/prtn17.jpg"]
    pic_num = 18
    focal_list = [704.916, 706.286, 705.849, 706.645, 706.587, 705.645, 705.327, 704.696, 703.794, 704.325, 704.696, 703.895, 704.289, 704.676, 704.847, 704.537, 705.102, 705.576]
    if len(sys.argv)==3:
        pic_num = 14

if len(sys.argv) == 2 and sys.argv[1] == "campus":
    files = ["../data/P1100277.jpg", "../data/P1100276.jpg", "../data/P1100275.jpg", "../data/P1100274.jpg", "../data/P1100273.jpg"]
    pic_num = 5
    focal_list = [505.767, 505.432, 505.307, 504.46, 504.662]

imgs = [cv2.imread(files[i]) for i in range(pic_num)]



def getfeature(I):
    # gaussian filter
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_f = np.exp(-(x**2+y**2))
    gaussian_f = gaussian_f / gaussian_f.sum()
    # gradient filter
    gradient_r_f = [[-1/4, 0, 1/4], [-1/2, 0, 1/2], [-1/4, 0, 1/4]]
    gradient_c_f = [[1/4, 1/2, 1/4], [0, 0, 0], [-1/4, -1/2, -1/4]]

    h, w = I.shape
    level = 4 # level of multi-scale harris corner detector
    feature = []
    f_list = []
    point_num = 500 # feature num in an img
    loc_r = [] # candidate locations
    
    for l in range(level):
        h_I, w_I = I.shape
        
        # gaussian
        gaussian_I = signal.convolve2d(I, gaussian_f, mode='same')
        # gradient
        I_x = signal.convolve2d(gaussian_I, gradient_r_f, mode='same')
        I_y = signal.convolve2d(gaussian_I, gradient_c_f, mode='same')
        S_xx = signal.convolve2d(I_x*I_x, gaussian_f, mode='same')
        S_yy = signal.convolve2d(I_y*I_y, gaussian_f, mode='same')
        S_xy = signal.convolve2d(I_x*I_y, gaussian_f, mode='same')
        
        np.seterr(divide='ignore',invalid='ignore')
        det = S_xx*S_yy - S_xy*S_xy
        trace = S_xx + S_yy
        f = det/trace    
        
        # candidate locations
        candidate_min = 10
        candidate_loc = []
        c_robust = 0.9
        
        f_candi_map = np.zeros((h_I, w_I), np.float)  
        b = 30
        for i in range(int(b/(2**l)), int(h_I-b/(2**l))):
            for j in range(int(b/(2**l)), int(w_I-b/(2**l))):
                if f[i][j]>10 and f[i][j] == np.max(f[i:i+4, j:j+4]):
                    candidate_loc.append([i, j])
                    f_candi_map[i][j] = f[i][j]
        
        # non-maximal supression
        for loc in candidate_loc:
            f_loc = f[loc[0]][loc[1]]
            tmp_list = np.argwhere(c_robust * f_candi_map > f_loc)
            if len(tmp_list) == 0:
                continue
            dis = (tmp_list[:, 0]-loc[0])**2 + (tmp_list[:, 1]-loc[1])**2
            min_dis = np.min(dis)
            loc_r.append([loc[0]*(2**l), loc[1]*(2**l), min_dis*(l+1)])

        loc_r.sort(key = lambda s: s[2], reverse = True)
        
        # make img of the next level
        h_I = int(h_I/2)
        w_I = int(w_I/2)
        tmp_I = np.zeros((h_I, w_I), np.float)
        for i in range(h_I):
            for j in range(w_I):
                tmp_I[i][j] = gaussian_I[i*2][j*2]
        I = tmp_I
    
    # get point_num features
    for i in range(point_num):
        f_list.append([loc_r[i][1], loc_r[i][0]])
    #coords = np.array(f_list)
    return np.array(f_list)


features = []
for id in range(pic_num):
    gray_img = cv2.imread(files[id], cv2.IMREAD_GRAYSCALE)
    features.append(getfeature(gray_img))


# MSOP descriptor

# gaussian filter
x, y = np.mgrid[-1:2, -1:2]
gaussian_f = np.exp(-(x**2+y**2))
gaussian_f = gaussian_f / gaussian_f.sum()
# gradient filter
gradient_r_f = [[-1/4, 0, 1/4], [-1/2, 0, 1/2], [-1/4, 0, 1/4]]
gradient_c_f = [[1/4, 1/2, 1/4], [0, 0, 0], [-1/4, -1/2, -1/4]]

descs = []
# descriptor
for pic_id in range(pic_num):
    descriptor_id = []
    I = cv2.cvtColor(imgs[pic_id], cv2.COLOR_BGR2GRAY)
    h, w = I.shape
    # get orientation
    gaussian_I = signal.convolve2d(I, gaussian_f, mode='same')
    g_r = signal.convolve2d(gaussian_I, gradient_r_f, mode='same')
    g_c = signal.convolve2d(gaussian_I, gradient_c_f, mode='same')

    # descriptors
    for f in features[pic_id]:
        d = []
        a = g_c[int(f[1])][int(f[0])]
        b = g_r[int(f[1])][int(f[0])]
        theta = math.atan(a/b)
        
        f_window = np.zeros((40, 40) ,np.float)
        for i in range(-20, 20):
            for j in range(-20, 20):
                new_i = int(i*math.cos(theta) - j*math.sin(theta) + f[1])
                new_j = int(i*math.sin(theta) - j*math.cos(theta) + f[0])
                tmp = I[new_i][new_j]
                f_window[i+20][j+20] = tmp
        for i in range(8):
            for j in range(8):
                d.append(np.mean(f_window[i*5:i*5+5, j*5:j*5+5]))
        # normalize
        d_std = np.std(d)
        d_mean = np.mean(d)
        d = (d-d_mean)/d_std
        descriptor_id.append(d)
    descs.append(descriptor_id)




# Cylindrical projection
h, w, c = imgs[0].shape
c_imgs = np.zeros((pic_num, h, w, c), np.uint8) # warped imgs
c_g = np.zeros((pic_num, h, w), np.float) # weight of warped imgs
c_features = [] # feature loc of warped imgs

g = np.zeros((h, w), np.float)
for i in range(h):
    for j in range(w):
        g[i][j] = (i-(h/2))**2 + (j-(w/2))**2


for id in range(pic_num):
    focal = focal_list[id]
    for y in range(-int(h/2), int(h/2)):
        for x in range(-int(w/2), int(w/2)):
            cylin_x = focal*math.atan(x/focal)
            cylin_y = focal * y / math.sqrt(x**2 + focal**2)

            cylin_x = int(round(cylin_x + w/2))
            cylin_y = int(round(cylin_y + h/2))
            if cylin_x >= 0 and cylin_x < w and cylin_y >= 0 and cylin_y < h:
                c_imgs[id][cylin_y][cylin_x] = imgs[id][y+int(h/2)][x+int(w/2)]
                c_g[id][cylin_y][cylin_x] = g[y+int(h/2)][x+int(w/2)]

    # feature
    c_f = []
    for f in features[id]:
        y = f[1]-int(h/2)
        x = f[0]-int(w/2)
        cylin_x = focal*math.atan(x/focal)
        cylin_y = focal*y/math.sqrt(x**2+focal**2)

        cylin_x = int(round(cylin_x + w/2))
        cylin_y = int(round(cylin_y + h/2))
        c_f.append([cylin_x, cylin_y])
    c_features.append(np.array(c_f))
    #cv2.imwrite("mywarp"+str(id)+".jpg", c_imgs[id])


# matches

inlier_list = []

match_list = []
thres = [0.9, 0.7, 0.9, 0.7]

point_num = 500
for pic_id in range(1, pic_num):

    ratio_threshold = thres[pic_id-1] if sys.argv[1]=='campus' else 0.55
    p1 = pic_id-1
    p2 = pic_id
    dists = scipy.spatial.distance.cdist(descs[p1], descs[p2])
    sort_idx = np.argsort(dists, 1)
    best_idx = sort_idx[:, 0]
    second_idx = sort_idx[:, 1]
    best_d = np.zeros((point_num, 1), np.float)
    second_d = np.zeros((point_num, 1), np.float)
    min_d = float("inf")
    for id in range(point_num):
        best_d[id] = dists[id][best_idx[id]]
        second_d[id] = dists[id][second_idx[id]]
        min_d = best_d[id] if best_d[id] < min_d else min_d
    ratio = best_d/second_d
    for i in range(len(best_d)):
        ratio[i] = float("inf") if best_d[i] > min_d*10 else ratio[i]
    match_id = np.argwhere(ratio < ratio_threshold)[:, 0]
    #match_id = match_id[:][0]
    match_id2 = best_idx[match_id].flatten()
    matches = np.array(np.c_[match_id, match_id2])
    #print (matches.shape)
    match_list.append(matches)






# rot
def rotate_imgs(warp_imgs ,g_imgs):
    rot_imgs = []
    new_g_imgs = []
    #print(warp_imgs.shape)
    theta = -math.pi/36
    for id in range(5):
        if id >0:
            rot_imgs.append(warp_imgs[id])
            new_g_imgs.append(g_imgs[id])
            continue
        h, w, c = warp_imgs[id].shape
        rot_img = np.zeros_like(warp_imgs[id])
        new_g_img = np.zeros_like(g_imgs[id])
        for i in range(h):
            for j in range(w):
                new_j = int(cos(theta)*(j-int(w/2)) - sin(theta)*(i-int(h/2))+int(w/2))
                new_i = int(sin(theta)*(j-int(w/2)) + cos(theta)*(i-int(h/2))+int(h/2))
                if new_j<w and new_j > 0 and new_i<h and new_i>0:
                    rot_img[new_i][new_j][:] = warp_imgs[id][i][j][:]
                    new_g_img[new_i][new_j] = g_imgs[id][i][j]
        for i in range(1, h-1):
            for j in range(1, w-1):
                if rot_img[i][j][0] == 0:
                    for c in range(3):
                        tot = [0, 0, 0]
                        for k in range(-1, 2):
                            for l in range(-1, 2):
                                    tot[c] += rot_img[i+k][j+l][c]
                        rot_img[i][j][c] = int(tot[c]/8)
                    new_g_img[i][j] = 1
               
        rot_imgs.append(rot_img)
        new_g_imgs.append(new_g_img)
        cv2.imwrite("rot_test.jpg", rot_img)
    
    return (np.array(rot_imgs), np.array(new_g_imgs))

if len(sys.argv) == 2:
    shifts = []
    s_list = [0.7, 0.5, 0.7, 0.5]
    for i in range(pic_num-1):
        matches = match_list[i]
        m1 = c_features[i][matches[:, 0], :]
        m2 = c_features[i+1][matches[:, 1], :]
        correct_match = np.hstack((m1, m2))

        #threshold=5#0.5#10 for good
        max_iterations=100
        correct_match = np.matrix(correct_match)

        iterations = 0
        max_cnt = 0
        h, w, c = imgs[0].shape
        # if we reached the maximum iteration
        x_m1 = m1[:, 0]
        y_m1 = m1[:, 1]
        x_m2 = m2[:, 0]
        y_m2 = m2[:, 1]
        max_cnt = 0
        max_id = None
        thres = 100
        s = s_list[i] if sys.argv[1]=='campus' else 0.5
        #s = 0.5
        for j in range(len(matches)):
            diff = m2[j] - m1[j]
            shift_x_m1 = x_m1 + diff[0]
            shift_y_m1 = y_m1 + diff[1]
            diff_x = shift_x_m1 - x_m2
            diff_y = shift_y_m1 - y_m2
            cnt = 0

            if diff[0]<(s*w):
                continue
            for k in range(len(matches)):
                if (sqrt((diff_x[k]**2).sum() + (diff_y[k]**2).sum()) < thres):
                    cnt +=1
                if not cnt < max_cnt:
                    max_cnt = cnt
                    max_id = j
            
        shift = m2[max_id]-m1[max_id]
        print(shift)
        shifts.append(shift)




    if sys.argv[1]=="campus":
        c_imgs, c_g = rotate_imgs(c_imgs, c_g)


    # stich

    h, w, c= imgs[0].shape
    new_img = np.zeros((h*2, w*(pic_num+1), 3), np.float)
    g = np.zeros((h*2, w*(pic_num+1)), float)
    s_y_min = h
    s_y_max = 0

    for id in range(pic_num):
        shift = [0, int(h/3)]
        for i in range(id):
            shift += shifts[pic_num-i-2]

        new_img[shift[1]:shift[1]+h, shift[0]:shift[0]+w][:, :, 0] += c_g[pic_num-id-1]*c_imgs[pic_num-id-1][:, :, 0]
        new_img[shift[1]:shift[1]+h, shift[0]:shift[0]+w][:, :, 1] += c_g[pic_num-id-1]*c_imgs[pic_num-id-1][:, :, 1]
        new_img[shift[1]:shift[1]+h, shift[0]:shift[0]+w][:, :, 2] += c_g[pic_num-id-1]*c_imgs[pic_num-id-1][:, :, 2]
        g[shift[1]:shift[1]+h, shift[0]:shift[0]+w] += c_g[pic_num-id-1]

        s_y_min = shift[1] if shift[1] < s_y_min else s_y_min
        s_y_max = shift[1] if shift[1] > s_y_max else s_y_max

    s_x_max = shift[0]+w
    new_img[:, :, 0] = new_img[:, :, 0]/g
    new_img[:, :, 1] = new_img[:, :, 1]/g
    new_img[:, :, 2] = new_img[:, :, 2]/g


    result = new_img[s_y_max:s_y_min+h, 0:s_x_max, :].astype(np.uint8)
    cv2.imwrite("result.png", result)

else:

    def homograph(x1, x2):
        # A
        A = np.zeros((8, 8), np.float)
        for i in range(4):
            A[i*2, :] = [x1[i, 0], x1[i, 1], 1, 0, 0, 0, -x2[i, 0] * x1[i, 0], -x2[i, 0] * x1[i, 1]]
            A[i*2+1, :] = [0, 0, 0, x1[i, 0], x1[i, 1], 1, -x2[i, 1] * x1[i, 0], -x2[i, 1] * x1[i, 1]]
        A = np.matrix(A)
        
        # B
        B = (x2.flatten()).reshape(8, 1)
        B = B.astype(float)
        
        # A*homo=B
        try:
            homo = np.linalg.solve(A, B)
        except:
            homo = np.linalg.lstsq(A, B)[0]
        return homo
    


    homos = [np.matrix(np.identity(3))]
    for i in range(pic_num-1):
        matches = match_list[i]
        m1 = features[i][matches[:, 0], :]
        m2 = features[i+1][matches[:, 1], :]
        correct_match = np.hstack((m1, m2))

        threshold=13
        max_iterations=500
        correct_match = np.matrix(correct_match)

        iterations = 0
        max_cnt = 0

        # if we reached the maximum iteration
        while iterations < max_iterations:

            # get 4 random matches and get homo 
            random_match = np.array(correct_match)
            np.random.shuffle(random_match)
            random_match_4 = random_match[:4]
            random_match_4 = np.matrix(random_match_4)
            random_match_4_p1 = random_match_4[:, :2]
            random_match_4_p2 = random_match_4[:, 2:]
            homo = homograph(random_match_4_p1, random_match_4_p2)
            homo = np.vstack((homo, np.matrix(1)))
            homo = homo.reshape((3, 3))
    
            # grab the match_2pics for the appropriate image
            correct_match_p1 = correct_match[:, :2].transpose()

            # third line
            # correct 
            ones = np.ones((1, correct_match_p1.shape[1]), np.float)
            correct_match_p1 = np.vstack((correct_match_p1, ones))
            correct_match_p2 = correct_match[:, 2:].transpose()
            # guess
            guess_match_p2 = (homo * correct_match_p1)
            ones = np.ones((1, guess_match_p2.shape[1]), np.float)
            guess_match_p2 = np.vstack((guess_match_p2, ones))[:2, :]
            
            # compute error
            error = np.sqrt((np.array(guess_match_p2 - correct_match_p2) ** 2).sum(0))
            #print(error)

            # inliner
            if (error < threshold).sum() > max_cnt:
                best_homo = homo
                max_cnt = (error < threshold).sum()

            iterations += 1

        homos.append(np.linalg.inv(best_homo))
    
    h_list = []
    for i in range(1, pic_num):
        homos[i] = homos[i - 1] * homos[i]
    for i in range(pic_num):
        h_list.append(np.linalg.inv(homos[pic_num//2])*homos[i])
    #print(h_list)
    #print(len(h_list))

    
    # get the size of global image: w_max-w_min * h_max-h_min
    I = cv2.imread(files[0])
    h, w, c = I.shape
    wlist = []
    hlist = []
    for id in range(pic_num):
        # corners: (0, 0), (w, 0), (w, h), (0, h)
        A = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]], np.float)
        homo = h_list[id]
        B_tmp = np.dot(homo, A)
        B = np.zeros_like(B_tmp)
        for i in range(3):
            B[i, :] = B_tmp[i, :] / B_tmp[2, :]
        #w1, w2 = int(B[0, 0]), int(B[0, 1])
        #h1, h2 = int(B[1, 0]), int(B[1, 1])"
        wlist.append(int(B[0, :].min()))
        wlist.append(int(B[0, :].max()))
        hlist.append(int(B[1, :].min()))
        hlist.append(int(B[1, :].max()))

    w_min = min(wlist)
    w_max = max(wlist)
    h_min = min(hlist)
    h_max = max(hlist)
    global_size = np.array([[w_min, h_min], [w_max, h_max]])

    # generate trans_imgs and gs
    trans_imgs = []
    trans_gs = []
    for id in range(pic_num):
        I =cv2.imread(files[id])
        h, w, c = I.shape

        y, x = np.mgrid[0:h, 0:w]
        y_trans, x_trans = np.mgrid[0:h, 0:w]
        g = (y-h/2)**2 + (x-w/2)**2

        y_trans, x_trans = np.mgrid[global_size[0, 1]:global_size[1, 1], global_size[0, 0]:global_size[1, 0]]
        h = global_size[1, 1] - global_size[0, 1]
        w = global_size[1, 0] - global_size[0, 0]

        # get A from B
        ones = np.ones((1, h*w), np.uint8)
        B = np.vstack((x_trans.flatten(), y_trans.flatten(), ones))
        A = np.dot(np.linalg.inv(np.array(h_list[id])), B)
        
        for i in range(3):
            A[i, :] = A[i, :] / A[2, :]

        A_x = A[0, :].reshape((h, w))
        A_y = A[1, :].reshape((h, w))

        trans_img = np.zeros((h, w, 3), np.int16)
        trans_g = np.zeros((h, w), np.float)
        #print("sizeI")

        for i in range(3):
            trans_img[..., i] = interpolation.map_coordinates(I[..., i], [A_y, A_x])
        trans_g = interpolation.map_coordinates(I[..., 3], [A_y, A_x])

        trans_imgs.append(trans_img)
        trans_gs.append(trans_g)
    

    img = np.zeros((h, w, 4), np.float)
    img_g = np.zeros((h, w, 4), np.float)
    for id in range(pic_num):
        for j in range(3):
            img[:, :, j] += trans_gs[id] * trans_imgs[id][:, :, j]
            img_g[:, :, j] += trans_gs[id]

    result = (img / img_g).astype(uint8)
    cv2.imwrite('result.jpg', result)
