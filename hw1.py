from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QSpinBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import re
import sys
import cv2
import numpy as np
import os
import torch 
import torchvision
import torchsummary
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision import models
import PIL
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('hw1.ui', self)
        self.files = []
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.w = 11
        self.h = 8
        self.obj_point = np.zeros((self.w*self.h, 3), dtype=np.float32)
        self.obj_point[:, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2)
        self.matrix = None
        self.matrix2 = None

        #Q4
        self.load_filename1 = ""
        self.load_filename2 = ""
        
        self.show()
        self.Load_folder.clicked.connect(self.load_folder_on_click)  
        self.Load_Image_L.clicked.connect(self.load_Lclick)
        self.Load_Image_R.clicked.connect(self.loadr_click)
        self.Find_corners.clicked.connect(self.Find_corners_click)
        self.Find_intrinsic.clicked.connect(self.Find_intrinsic_click)
        self.spinBox.valueChanged.connect(self.spinbox)
        self.Find_extrinsic.clicked.connect(self.Find_extrinsic_click)
        self.Find_distortion.clicked.connect(self.Find_distortion_click)
        self.Show_result.clicked.connect(self.Show_result_click)
        self.Show_word_horizon.clicked.connect(self.Show_horizon_click)
        self.show_word_vertical.clicked.connect(self.Show_vertical_click)
        self.stereo_disparity_map.clicked.connect(self.stereo_disparity_map_click)
        self.Load_Image1.clicked.connect(self.load_image1_click)
        self.Load_Image2.clicked.connect(self.load_image2_click)
        self.keypoints.clicked.connect(self.keypoints_click)
        self.match_keypoint.clicked.connect(self.match_click)
        self.Show_Agumented_Image.clicked.connect(self.Agument_Image_clicked)
        self.Show_Model_Structure.clicked.connect(self.Show_Model_click)
        self.Show_Acc_and_Loss.clicked.connect(self.Show_Accuracy_Loss_click)
        self.Inference.clicked.connect(self.Inference_click)
        self.Load_Image_VGG.clicked.connect(self.load_Q5_click)


    #LOAD image 
    def load_folder_on_click(self):
        self.load_all_dir_name_file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        overall = [i for i in os.listdir(self.load_all_dir_name_file) if i.endswith('.bmp')]
        self.files = sorted(overall, key=lambda x: int(x.split('.')[0]))
        print(self.files)
    
    def load_Lclick(self):
        self.l = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
        print(self.l)
    
    def loadr_click(self):
        self.r = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
        print(self.r)
    

    def Find_corners_click(self):
        point_3d = []
        point_2d = []
        for file in self.files:
            img = cv2.imread(os.path.join(self.load_all_dir_name_file, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            _,corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
            #corners = cv2.UMat(corners)
            if corners is not None:
                new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), self.criteria)
                img_out = cv2.drawChessboardCorners(img, (self.w, self.h), new_corners, True)
                point_3d.append(self.obj_point)
                point_2d.append(new_corners)
                cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Corners", 1024, 1024)
                cv2.imshow('Corners', img_out)
                cv2.waitKey(100)
        self.matrix = cv2.calibrateCamera(point_3d, point_2d, gray.shape[::-1], None, None)
        
        cv2.destroyAllWindows()

    def Find_intrinsic_click(self):
            print("Instrinsic:", self.matrix[1], sep = '\n')
    
    def spinbox(self):
            print(self.spinBox.value())
    
    def Find_extrinsic_click(self):
        index = int(self.spinBox.value()) - 1
        rotation_mat= cv2.Rodrigues(self.matrix[3][index])[0]
        extrinsic_mat = np.hstack([rotation_mat, self.matrix[4][index]])
        print("Extrinsic", extrinsic_mat, sep="\n")

    def Find_distortion_click(self):
         print("Intrinsic:", self.matrix[2], sep="\n")

    def Show_result_click(self):
        for file in self.files:
                img = cv2.imread(os.path.join(self.load_all_dir_name_file, file))
                h, w = img.shape[:2]
                newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(self.matrix[1], self.matrix[2], (w, h), 0, (w, h))
                dst = cv2.undistort(img, self.matrix[1], self.matrix[2], None, newcameramatrix)

                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                img = cv2.resize(img, (480,480))
                dst = cv2.resize(dst, (480,480))
                imgs = np.hstack([dst, img])
                cv2.imshow("undistorted result", imgs)
                cv2.waitKey(600)
        cv2.destroyAllWindows()
    
    
    def calibaration2(self):
        point_3d = []
        point_2d = []
        for file in self.files:
            img = cv2.imread(os.path.join(self.load_all_dir_name_file, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
            if(ret == True):
                new_corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), self.criteria)
                point_3d.append(self.obj_point)
                point_2d.append(new_corners)
        # ret, intrinsic, distort, r_vecs, t_vecs
        self.matrix2 = cv2.calibrateCamera(point_3d, point_2d, gray.shape[::-1], None, None)

    def draw(self, img, corners, img_point, len):  
        img_point = np.int32(img_point).reshape(-1,2)
        for i in range(len):
            img = cv2.line(img, tuple(img_point[2*i]), tuple(img_point[2*i+1]),(0, 0, 255), 15)
        return img 
    
    def Show_horizon_click(self):
        self.calibaration2()

        word = []
        text = self.textEdit.toPlainText().upper()
        lib = os.path.join(self.load_all_dir_name_file,'Q2_lib/alphabet_lib_onboard.txt')
        fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

        length = 0
        for i in range(len(text)):
            if text[i].encode('UTF-8').isalpha() and not text[i].isdigit():
                word.append(fs.getNode(text[i]).mat())
                length = length + 1

        pos_adjust=[[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]

        for i in range(length):
            for j in range(len(word[i])):
                new_axis1 = [a + b for a, b in zip(word[i][j][0], pos_adjust[i])]
                new_axis2 = [a + b for a, b in zip(word[i][j][1], pos_adjust[i])]
                word[i][j][0]=new_axis1
                word[i][j][1]=new_axis2

        for i in range(len(self.files)):
            img = cv2.imread(os.path.join(self.load_all_dir_name_file, self.files[i]))
            rotation_vector= cv2.Rodrigues(self.matrix2[3][i])[0]
            transform_vector = self.matrix2[4][i]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
            new_corners = cv2.cornerSubPix(gray, corners, (15,15),(-1,-1), self.criteria)
            
            # ret, intrinsic, distort, r_vecs, t_vecs
            axis = []
            for j in range(len(word)):
                axis1 = np.array(word[j], dtype=np.float32).reshape(-1,3)
                axis.append(axis1)
                img_points, jac = cv2.projectPoints(axis[j], rotation_vector, transform_vector, self.matrix2[1], self.matrix2[2])
                img =  self.draw(img, new_corners, img_points, len(word[j]))
            
            cv2.namedWindow('Augmented Reality', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Augmented Reality", 480, 480)
            cv2.imshow('Augmented Reality',img)
            cv2.waitKey(500)
        cv2.destroyAllWindows()
        
    def Show_vertical_click(self):
        self.calibaration2()

        word = []
        text = self.textEdit.toPlainText().upper()

        lib = os.path.join(self.load_all_dir_name_file,'Q2_lib/alphabet_lib_vertical.txt')
        fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

        length = 0
        for i in range(len(text)):
            if text[i].encode('UTF-8').isalpha() and not text[i].isdigit():
                word.append(fs.getNode(text[i]).mat())
                length = length + 1

        pos_adjust=[[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]

        for i in range(length):
            for j in range(len(word[i])):
                new_axis1 = [a + b for a, b in zip(word[i][j][0], pos_adjust[i])]
                new_axis2 = [a + b for a, b in zip(word[i][j][1], pos_adjust[i])]
                word[i][j][0]=new_axis1
                word[i][j][1]=new_axis2

        for i in range(len(self.files)):
            img = cv2.imread(os.path.join(self.load_all_dir_name_file, self.files[i]))
            rotation_vector= cv2.Rodrigues(self.matrix2[3][i])[0]
            transform_vector = self.matrix2[4][i]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
            new_corners = cv2.cornerSubPix(gray, corners, (15,15),(-1,-1), self.criteria)
            
            # ret, intrinsic, distort, r_vecs, t_vecs
            axis = []
            for j in range(len(word)):
                axis1 = np.array(word[j], dtype=np.float32).reshape(-1,3)
                axis.append(axis1)
                img_points, jac = cv2.projectPoints(axis[j], rotation_vector, transform_vector, self.matrix2[1], self.matrix2[2])
                img =  self.draw(img, new_corners, img_points, len(word[j]))
            
            cv2.namedWindow('AR vertical', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AR vertical", 480, 480)
            cv2.imshow('AR vertical',img)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def stereo_disparity_map_click(self):
        if self.l and self.r:
            imgl = cv2.imread(self.l)
            imgr = cv2.imread(self.r)
            imgl_gray = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
            imgr_gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)

            stereo = cv2.StereoBM_create(numDisparities=21 * 16, blockSize = 19)
            disparity = stereo.compute(imgl_gray, imgr_gray)
            disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disparity_show = cv2.cvtColor(disparity,cv2.COLOR_GRAY2RGB)
            cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
            cv2.resizeWindow("disparity", 720, 540)
            cv2.imshow("disparity", disparity_show)

            def mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    imgr_cp = imgr.copy()
                    imgl_cp = imgl.copy()
                    if disparity[y, x] != 0:
                        cv2.circle(imgr_cp, (x-disparity[y, x]-50, y), 8, (255,255,0), thickness = -1)
                        print('disparity: {}'.format(disparity[y, x]))
                        print('depth: {}'.format(int(342.789*4019.284/(279.184+disparity[y, x]))))
                        print(" (" , x, y , ")")
                        cv2.imshow("imgr", imgr_cp)
                        cv2.imshow("imgl", imgl_cp)

            cv2.namedWindow('imgl',cv2.WINDOW_NORMAL)
            cv2.resizeWindow("imgl", 720, 540)
            cv2.namedWindow('imgr',cv2.WINDOW_NORMAL)
            cv2.resizeWindow("imgr", 720, 540)
            cv2.imshow("imgl", imgl)
            cv2.imshow("imgr", imgr)
            cv2.setMouseCallback("imgl", mouse)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
    
    def load_image1_click(self):
        self.load_filename1 = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
        print(self.load_filename1)

    def load_image2_click(self):
        self.load_filename2 = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
        print(self.load_filename2)

    def keypoints_click(self):
            if self.load_filename1 == "":
                return
            img1 = cv2.imread(self.load_filename1)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            SIFT = cv2.SIFT_create()
            key1, des1 = SIFT.detectAndCompute(img1_gray, None)

            kp_image1 = cv2.drawKeypoints(img1_gray, key1, np.array([]), color=(0,255,0))
            cv2.imshow("Keypoints", kp_image1)

            cv2.waitKey()
            cv2.destroyAllWindows()
    
    def match_click(self):
        if self.load_filename1 == "" or self.load_filename2 == "":
            return
        MIN_MATCH_COUNT = 10

        img1 = cv2.imread(self.load_filename1)
        img2 = cv2.imread(self.load_filename2)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        SIFT = cv2.SIFT_create() #change
        key1, des1 = SIFT.detectAndCompute(img1_gray, None)
        key2, des2 = SIFT.detectAndCompute(img2_gray, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    
        goodMatches = []
        minRatio = 0.75
        for m,n in matches:
            if m.distance < minRatio * n.distance:
                goodMatches.append([m])

        img3 = cv2.drawMatchesKnn(img1_gray, key1, img2_gray, key2, goodMatches, outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        
        cv2.namedWindow('Matched', cv2.WINDOW_NORMAL,0)
        cv2.resizeWindow("Matched", 640, 640)
        cv2.imshow('Matched',img3)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def Agument_Image_clicked(self):
        self.Q5_1_dir = ".\Dataset_CvDl_Hw1\Q5_image\Q5_1"
        self.Q5_1_files = [i for i in os.listdir(self.Q5_1_dir)]

        col_nums = 3
        rows_nums = 3
        i = 1
        transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(30),
            ])
        fig = plt.figure(figsize=(30, 15))

        for file in self.Q5_1_files:
            img = PIL.Image.open(os.path.join(self.Q5_1_dir,file))
            img = transform(img)
            img_array = plt.np.array(img)
            ax = fig.add_subplot(rows_nums, col_nums, i)
            i=i+1
            ax.imshow(img_array)
            ax.set_title(file)
        plt.tight_layout()
        plt.show()
    
    def load_Q5_click(self):
        self.load_Q5_4 = str(QFileDialog.getOpenFileName(self, "Choose a file")[0])
        print(self.load_Q5_4)
        if self.load_Q5_4 == "":
            return
        original_pixmap = QPixmap(self.load_Q5_4)
        scaled_pixmap = original_pixmap.scaled(128, 128)

        self.label.setPixmap(scaled_pixmap)
    
    def Show_Model_click(self):
        model = models.vgg19_bn(num_classes = 10)
        torchsummary.summary(model,input_size=(3,32,32),device='cpu')

    def Show_Accuracy_Loss_click(self):
        img = cv2.imread('loss_plot.png')
        cv2.namedWindow('Accuracy_Loss', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Accuracy_Loss", 1600, 900)
        cv2.imshow('Accuracy_Loss',img)
    
    def Inference_click(self):
        if self.Load_Image_VGG == "":
            print("no image load")
            return
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


        model = models.vgg19_bn(pretrained=False,num_classes = 10)
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()

        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img=PIL.Image.open(self.Load_Image_VGG)
        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0)

        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

        test = "predict = " + str(classes[predicted_class])
        
        self.label.setText(test)

        plt.bar(range(10), probabilities.detach().numpy())        
        
        plt.xlabel("class")
        plt.xticks(range(len(classes)), classes)

        plt.ylabel("probabilities")
        plt.title("probabilities distribution")
        plt.show()




app = QtWidgets.QApplication(sys.argv)
window = Ui()
window.show()
app.exec_()