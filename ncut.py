import numpy as np
import cv2
import os
from scipy.linalg import eigh
from math import sin,cos

sigmaI = 20
sigmaX = 30
r = 10

def Fi_BGR(ix,iy,jx,jy,img):
    return img[ix,iy]-img[jx,jy]

def Fi_HSV(ix,iy,jx,jy,img):
    hsv_img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2HSV)
    h = hsv_img[:,:,0]
    s = hsv_img[:,:,1]
    v = hsv_img[:,:,2]

    hi = h[ix,iy]
    si = s[ix,iy]
    vi = v[ix,iy]

    hj = h[jx,jy]
    sj = s[jx,jy]
    vj = v[jx,jy]
    
    Fi = np.array([vi,vi*si*sin(hi),vi*si*cos(hi)])
    Fj = np.array([vj,vj*sj*sin(hj),vj*sj*cos(hj)])

    return Fi-Fj

def feature_sim(ix,iy,jx,jy,img,choice):
    if choice == 0:
        diff = Fi_HSV(ix,iy,jx,jy,img)
    else:
        diff = Fi_BGR(ix,iy,jx,jy,img)
    norm = diff @ diff
    return np.exp(-norm/sigmaI)


    
def spatial_sim(ix,iy,jx,jy):
    dist = (ix-jx)**2 + (iy-jy)**2
    if dist > r:
        return 0
    return np.exp(-dist/sigmaX)

def sim(img,i,j,choice):
    (h,w,_) = img.shape
    ix = i//h
    iy = i%h

    jx = j//h
    jy = j%h

    return feature_sim(ix,iy,jx,jy,img,choice) * spatial_sim(ix,iy,jx,jy)

def Gen_EV(A,B,thresh):
    # Ax = LBx
    eigvals, eigvecs = eigh(A, B, eigvals_only=False, subset_by_index=[0,100])
    #print(eigvals)
    for i in range(100):
        if eigvals[i]>thresh:
            return eigvecs[:,i]
    return eigvecs[:,99]



img_path = 'test_images/test3.jpg'
img = cv2.imread(img_path)

def Ncut(img,choice):
    (h,w,_) = img.shape
    if(h>100):
        img = cv2.resize(img,(100,100))
    #img = normalize_meanstd(img, axis=(1,2))

    (h,w,_) = img.shape
    n = w*h
    W = np.zeros((n,n))
    D = np.zeros((n,n))
    for i in range(n):
        if(i%1000 == 0):
            print(i)
        for j in range(i+1):
            W[i,j] = sim(img,i,j,choice)
            #print(W[i,j])
            W[j,i] = W[i,j]
    for i in range(n):
        D[i,i] = np.sum(W[i,:])

    print("W--Done")

    eigen_vector = Gen_EV(D-W,D,1e-4)

    print("Eig--Done")

    seg = np.zeros((h,w))

    T = 0
    for i in range(h):
        for j in range(w):
            if eigen_vector[i*h+j]>T:
                seg[i,j] = 255

    print("Seg--Done")
    return seg


folder = r'C:\Users\bandl\Desktop\MTech_IISc_CSA\Sem_2\AIP\Assignments\Assignment_2\test_images'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    print(filename)
    if img is not None:
        seg = Ncut(img,1)
        cv2.imwrite('ncut_images/'+filename.split(".")[0]+'/'+'Original_Fi_BGR.jpg', seg)
        
        seg = Ncut(img,0)
        cv2.imwrite('ncut_images/'+filename.split(".")[0]+'/'+'Original_Fi_HSV.jpg', seg)

        seg = Ncut(cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE),1)
        cv2.imwrite('ncut_images/'+filename.split(".")[0]+'/'+'Rotated_Fi_BGR.jpg', seg)

        seg = Ncut(cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE),0)
        cv2.imwrite('ncut_images/'+filename.split(".")[0]+'/'+'Rotated_Fi_HSV.jpg', seg)

        guass_noise = np.random.normal(scale = 3,size = img.shape)
        seg = Ncut(img+guass_noise,1)
        cv2.imwrite('ncut_images/'+filename.split(".")[0]+'/'+'Noisy_Fi_BGR.jpg', seg)

        seg = Ncut(img+guass_noise,0)
        cv2.imwrite('ncut_images/'+filename.split(".")[0]+'/'+'Noisy_Fi_HSV.jpg', seg)
