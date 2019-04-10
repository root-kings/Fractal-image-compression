#imports for GUI
import sys
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from PyQt5 import QtGui,QtWidgets,QtCore
from PyQt5.QtWidgets import (QWidget, QLabel, 
    QComboBox, QApplication)
import os
#Imports for compression
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy import optimize
import numpy as np
import math
import PIL
from PIL import Image


#Photo angles and directions
directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = [[direction, angle] for direction in directions for angle in angles]

 
class App(QWidget):

    #Initializing function
    def __init__(self,parent=None):
        super(App,self).__init__(parent)
        self.title = 'Fractal Image Compression'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
        layout = QHBoxLayout()
        

        
        '''Radio buttons or quality selector'''
        
        #Radiobutton 1
        radiobutton = QRadioButton("256 by 256")
        radiobutton.quality = "256 by 256"
        radiobutton.toggled.connect(self.quality_select)
        layout.addWidget(radiobutton)
        #Radiobutton 2
        radiobutton = QRadioButton("512 by 512")
        radiobutton.quality = "512 by 512"
        radiobutton.toggled.connect(self.quality_select)
        layout.addWidget(radiobutton)
        #Radiobutton3
        radiobutton = QRadioButton("1024 by 1024")
        radiobutton.quality = "1024 by 1024"
        radiobutton.toggled.connect(self.quality_select)
        layout.addWidget(radiobutton)


        #Drop down
        self.cb = QComboBox()
        
        self.cb.addItem("Select image")
        self.cb.addItem("GreyScale")
        self.cb.addItem("Colored")
        self.cb.currentIndexChanged.connect(self.selectionchange)
        self.cb.move(500,450)
            
        layout.addWidget(self.cb)
        self.setLayout(layout)

    
        
    
    #Main function
    def initUI(self):
        self.setWindowTitle("Fractal image compressor")
        self.setGeometry(self.left, self.top, self.width, self.height)

        lbl1 = QLabel('Fractal Image Compression', self)
        lbl1.move(10, 10)
        lbl2= QLabel('Choose quality of image',self)
        lbl2.move(10,200)
        lbl3= QLabel('Progress of compression',self)
        lbl3.move(290,65)
        
        button = QPushButton('Select image', self)
        button.setToolTip('This is an example button')
        button.resize(100,32)
        button.move(5,60)
        button.clicked.connect(self.openFileNameDialog)

        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(QApplication.instance().quit)
        qbtn.resize(100,32)
        qbtn.move(505,430) 

        

        #progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)
        self.pbar.move(430,60)

        self.btn = QPushButton('Compress', self)
        self.btn.resize(100,32)
        self.btn.move(400,430)
        self.btn.clicked.connect(self.on_click)

        self.timer = QBasicTimer()
        self.step = 0
        self.show()

    def timerEvent(self, e):
      
        if self.step >= 100:
            
            self.timer.stop()
            self.btn.setText('Finished')
            return
            
        self.step = self.step + 1
        self.pbar.setValue(self.step)
        


      
        
    

    #For opening file
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        global img_input
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Image", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            img_input = fileName
            print(fileName)
    
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            
            print("runned")
    

    #Quality selector
    def quality_select(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            global qualit
            qualit = radioButton.quality
            if qualit=="256 by 256":
                pix = 256
            elif qualit=="512 by 512":
                pix = 512
            elif qualit=="1024 by 1024":
                pix = 1024
            else:
                print('Wrong choice of quality')
            #resizing image  
            img = Image.open(img_input)
            wpercent = (pix / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((pix, hsize), PIL.Image.ANTIALIAS)
            hpercent = (pix / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            img = img.resize((wsize, pix), PIL.Image.ANTIALIAS)
            global name
            name = 'resized_image.jpg'
            img.save(name)
            print("Quality is %s" % (radioButton.quality))

    
    #To select imae type
    def selectionchange(self,i):
        global choice
        choice = self.cb.currentText() 
        print ("Pic type is",choice)        
    def on_click(self):
        '''Main backend of the
            desktop app fractal image compressor
            Contains compression functions and many other
        '''
        
        '''
        if self.timer.isActive():
            self.timer.stop()
            self.btn.setText('Start')
        else:
            self.timer.start(100, self)
            self.btn.setText('Stop')
        '''


        def get_greyscale_image(img):
            return np.mean(img[:,:,:2], 2)

        def extract_rgb(img):
            return img[:,:,0], img[:,:,1], img[:,:,2]

        def assemble_rbg(img_r, img_g, img_b):
            shape = (img_r.shape[0], img_r.shape[1], 1)
            return np.concatenate((np.reshape(img_r, shape), np.reshape(img_g, shape), 
                np.reshape(img_b, shape)), axis=2)

        # Transformations

        def reduce(img, factor):
            result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    result[i,j] = np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
            return result

        def rotate(img, angle):
            return ndimage.rotate(img, angle, reshape=False)

        def flip(img, direction):
            return img[::direction,:]

        def apply_transformation(img, direction, angle, contrast=1.0, brightness=0.0):
            return contrast*rotate(flip(img, direction), angle) + brightness

        # Contrast and brightness

        def find_contrast_and_brightness1(D, S):
            # Fix the contrast and only fit the brightness
            contrast = 0.75
            brightness = (np.sum(D - contrast*S)) / D.size
            return contrast, brightness 

        def find_contrast_and_brightness2(D, S):
            # Fit the contrast and the brightness
            A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
            b = np.reshape(D, (D.size,))
            x, _, _, _ = np.linalg.lstsq(A, b)
            #x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
            return x[1], x[0]

        # Compression for greyscale images

        def generate_all_transformed_blocks(img, source_size, destination_size, step):
            factor = source_size // destination_size
            transformed_blocks = []
            for k in range((img.shape[0] - source_size) // step + 1):
                for l in range((img.shape[1] - source_size) // step + 1):
                    # Extract the source block and reduce it to the shape of a destination block
                    S = reduce(img[k*step:k*step+source_size,l*step:l*step+source_size], factor)
                    # Generate all possible transformed blocks
                    for direction, angle in candidates:
                        transformed_blocks.append((k, l, direction, angle, apply_transformation(S, direction, angle)))
            return transformed_blocks

        def compress(img, source_size, destination_size, step):
            transformations = []
            transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step)
            i_count = img.shape[0] // destination_size
            j_count = img.shape[1] // destination_size
            for i in range(i_count):
                transformations.append([])
                for j in range(j_count):
                    print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
                    transformations[i].append(None)
                    min_d = float('inf')
                    # Extract the destination block
                    D = img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size]
                    # Test all possible transformations and take the best one
                    for k, l, direction, angle, S in transformed_blocks:
                        contrast, brightness = find_contrast_and_brightness2(D, S)
                        S = contrast*S + brightness
                        d = np.sum(np.square(D - S))
                        if d < min_d:
                            min_d = d
                            transformations[i][j] = (k, l, direction, angle, contrast, brightness)
            
            return transformations

        def decompress(transformations, source_size, destination_size, step, nb_iter=8):
            factor = source_size // destination_size
            height = len(transformations) * destination_size
            width = len(transformations[0]) * destination_size
            iterations = [np.random.randint(0, 256, (height, width))]
            cur_img = np.zeros((height, width))
            for i_iter in range(nb_iter):
                print(i_iter)
                for i in range(len(transformations)):
                    for j in range(len(transformations[i])):
                        # Apply transform
                        k, l, flip, angle, contrast, brightness = transformations[i][j]
                        S = reduce(iterations[-1][k*step:k*step+source_size,l*step:l*step+source_size], factor)
                        D = apply_transformation(S, flip, angle, contrast, brightness)
                        cur_img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size] = D
                iterations.append(cur_img)
                cur_img = np.zeros((height, width))
            return iterations

        # Compression for color images

        def reduce_rgb(img, factor):
            img_r, img_g, img_b = extract_rgb(img)
            img_r = reduce(img_r, factor)
            img_g = reduce(img_g, factor)
            img_b = reduce(img_b, factor)
            return assemble_rbg(img_r, img_g, img_b)

        def compress_rgb(img, source_size, destination_size, step):
            img_r, img_g, img_b = extract_rgb(img)
            return [compress(img_r, source_size, destination_size, step), \
                compress(img_g, source_size, destination_size, step), \
                compress(img_b, source_size, destination_size, step)]

        def decompress_rgb(transformations, source_size, destination_size, step, nb_iter=8):
            img_r = decompress(transformations[0], source_size, destination_size, step, nb_iter)[-1]
            img_g = decompress(transformations[1], source_size, destination_size, step, nb_iter)[-1]
            img_b = decompress(transformations[2], source_size, destination_size, step, nb_iter)[-1]
            return assemble_rbg(img_r, img_g, img_b)

        # Plot

        def plot_iterations(iterations, target=None):
            # Configure plot
            plt.figure()
            nb_row = math.ceil(np.sqrt(len(iterations)))
            nb_cols = nb_row
            # Plot
            for i, img in enumerate(iterations):
                plt.subplot(nb_row, nb_cols, i+1)
                plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
                if target is None:
                    plt.title(str(i))
                else:
                    # Display the RMSE
                    plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
                frame = plt.gca()
                frame.axes.get_xaxis().set_visible(False)
                frame.axes.get_yaxis().set_visible(False)
            plt.tight_layout()

        # Tests

        def test_greyscale():
            img = mpimg.imread(name)
            img = get_greyscale_image(img)
            img = reduce(img, 4)
            plt.figure()
            plt.imshow(img, cmap='gray', interpolation='none')
            transformations = compress(img, 8, 4, 8)
            iterations = decompress(transformations, 8, 4, 8)
            plot_iterations(iterations, img)
            plt.show()


            

        def test_rgb():
            img = mpimg.imread(name)
            img = reduce_rgb(img, 8)
            transformations = compress_rgb(img, 8, 4, 8)
            retrieved_img = decompress_rgb(transformations, 8, 4, 8)
            plt.figure()
            plt.subplot(121)
            plt.imshow(np.array(img).astype(np.uint8), interpolation='none')
            plt.subplot(122)
            plt.imshow(retrieved_img.astype(np.uint8), interpolation='none')
            plt.show()

        #Running the main function
        if choice == 'GreyScale':
            self.timer.start(100, self)
            test_greyscale()
        elif choice == 'Colored':
            self.timer.start(100, self)
            test_rgb()



#Executing function				
def main():
    
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

 
if __name__ == '__main__':
    main()
    