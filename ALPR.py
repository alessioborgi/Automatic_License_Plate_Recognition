#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 08:42:32 2022

@Author:     Alessio Borgi, Martina Doku, Giuseppina Iannotti
@Contact :   borgi.*******@studenti.uniroma1.it, doku.******@studenti.uniroma1.it, iannotti.*******@studenti.uniroma1.it

@Filename:   ALPR.py
"""

#IMPORTING LIBRARIES WE WILL USE IN THE PROGRAM:
import pandas as pd                                                 #It is used to visualize the dataset Employee(ID_Employee,Name_Employee,Surname_Employee).
import cv2                                                          #It is used for image processing. We need to perform some operations on our images(like coverting them
                                                                    #in grayscale,reducing the noise). It allows to recognize the license plate better. 
import numpy as np                                                  #It is used in various part of our program, thanks to its efficiency wrt Python lists.
import math                                                         #It is used to perform mathematical operations like Pythagorean Distance, Trigonometry_Angle, useful for 
                                                                    #detecting the characters of the license plate.
import datetime                                                     #It is used to display the current date and time.
import pickle as pkl                                                #It is used for serializing the Module given by the KNN into a stream of bytes. 
from sklearn.neighbors import KNeighborsClassifier as KNN_sklearn   #It is used to do the recognition of the license plate .
import torch.nn
import matplotlib.pyplot as plt                                     #Sometimes,It is used to see samples (look at comments in the code).
from collections import namedtuple                                  #It is used to declare the tuples with the corresponding brightness level and color,in the 
                                                                    #Get_Level_Brigthness_image function.
from torch import nn                                                #Importin neural network package for torch .
from torch.utils.data import DataLoader                             #DataLoader wraps an iterable around the Dataset to enable easy access to the samples. 
import torchvision.transforms.functional as TF                      #It is a module useful to build a complex transformation pipeline.
from torchvision.transforms import ToTensor                         #It is used several times in our program coverting lists to Tensors. 
from torch.utils.data import Dataset                                #Dataset stores the samples and their corresponding labels.
from torchvision import transforms                                  #Use transforms to perform some manipulation of the data and make it suitable for training.
from torchvision.io import read_image                               #It reads a JPEG or PNG image into a 3 dimensional RGB or grayscale Tensor.
import csv                                                          #It is used to write data in the Alpha_Numeric_Customized_Dataset.csv file.
import os                                                           #It is used in the Loading_and_Donwload_Dataset function.  
from os import walk                                                 #It is used in the Loading_and_Donwload_Dataset function.
import joblib                                                       #It is used to save the KNN module.


#DATABASE CONNECTION AND INITIALIZATION OF NEEDED VARIABLES:
import psycopg2                                                     #It is used for handling connection with the database. 
import psycopg2.extras                                              #It is used for handling connection with the database.
hostname = 'localhost'                                              #Local host of the databas.
database = 'Employee_Detection_ALPR_Project'                        #Name of the database in Postgres.
username = 'postgres'                                               #Username used in Postgres. 
pwd = 'admin'                                                       #Password used in Postgres.
port_id = 5432                                                      #Port used in Postgres. 
conn = None                                                         #Connection to Postgres.  

#SEVERAL CONSTANTS DECLARATION:
#Those will be constants for Candidate_Character_Check, this checks one possible char only (does not compare to another char.
#Setting parameters that will be used in our program. Note that, they have been set after some Empirical Derivations (we tried by-hanf what was their best value). :)

MIN_PIXEL_WIDTH = 2                         #It is used in the Candidate_Character_Check function, in order to check if a character can be a candidate character. 
MIN_PIXEL_HEIGHT = 8                        #It is used in the Candidate_Character_Check function, in order to check if a character can be a candidate character. 

MIN_ASPECT_RATIO = 0.25                     #It is used in the Candidate_Character_Check function, in order to check if a character can be a candidate character. 
MAX_ASPECT_RATIO = 1.0                      #It is used in the Candidate_Character_Check function, in order to check if a character can be a candidate character. 

MIN_PIXEL_AREA = 80                         #It is used in the Candidate_Character_Check function, in order to check if a character can be a candidate character. 

#Constants for comparing two chars:
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3           #It is used to detect possible overlapping characters (like 'O') in the license plate, checkining whether the pythagorean 
                                            #distance between characters is very tiny. 
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0           #It is used to check wheter the character we are currently looking at could be added to the list of possible mathing characters.
MAX_CHANGE_IN_AREA = 0.5                    #It is used to check wheter the character we are currently looking at could be added to the list of possible mathing characters.
MAX_CHANGE_IN_WIDTH = 0.8                   #It is used to check wheter the character we are currently looking at could be added to the list of possible mathing characters.
MAX_CHANGE_IN_HEIGHT = 0.2                  #It is used to check wheter the character we are currently looking at could be added to the list of possible mathing characters.
MAX_ANGLE_BETWEEN_CHARS = 12.0              #It is used to check wheter the character we are currently looking at could be added to the list of possible mathing characters.

#Other constants:
MIN_NUMBER_OF_MATCHING_CHARS = 3            #It is used to check if the lenght of the license plate is greater than 3, in order to be a "valid" license plate. 
RESIZED_CHAR_IMAGE_WIDTH = 28               #For each character in the License Plate, we crop out the character, creating a sort of ROI (Region of Interest).
RESIZED_CHAR_IMAGE_HEIGHT = 28              #For each character in the License Plate, we crop out the character, creating a sort of ROI (Region of Interest).

#Setting parameters that will be used in our program. Empirical derivation of the thresholds.
PLATE_WIDTH_PADDING_FACTOR = 1.3            #It is used to add some padding to the rectangle of the license plate, otherwise characters would be cropped out.  
PLATE_HEIGHT_PADDING_FACTOR = 1.5           #It is used to add some padding to the rectangle od thelicense plate, otherwise characters would be cropped out.


#DICTIONARY DEFINITIONS FOR HANDLING LABELS-ALPHANUMERIC RECOGNITIONS:
dict_labels_to_alpha = { 0:'0',    1:'1',      2:'2',      3:'3',      4:'4',      5:'5',      6:'6',      7:'7',     8:'8',       9:'9',
                        10:'A',   11:'B',     12:'C',     13:'D',     14:'E',     15:'F',     16:'G',     17:'H',    18:'I',      19:'J',
                        20:'K',   21:'L',     22:'M',     23:'N',     24:'O',     25:'P',     26:'Q',     27:'R',    28:'S',      29:'T',
                        30:'U',   31:'V',     32:'W',     33:'X',     34:'Y',     35:'Z',     36:'a',     37:'b',    38:'c',      39:'d',
                        40:'e',   41:'f',     33:'g',     43:'h',     44:'i',     45:'j',     46:'k',     47:'l',    48:'m',      49:'n',
                        50:'o',   51:'p',     52:'q',     53:'r',     54:'s',     55:'t',     55:'u',     57:'v',    58:'w',      59:'x',
                        51:'y',   61:'z'}
dict_alpha_to_labels= { '0':0,   '1':1,      '2':2,      '3':3,      '4':4,      '5':5,      '6':6,      '7':7,     '8':8,       '9':9,
                        'A':10,   'B':11,     'C':12,     'D':13,     'E':14,     'F':15,    'G':16,     'H':17,    'I':18,      'J':19,
                        'K':20,   'L':21,     'M':22,     'N':23,     'O':24,     'P':25,     'Q':26,     'R':27,    'S':28,     'T':29,
                        'U':30,   'V':31,     'W':32,     'X':33,     'Y':34,     'Z':35}


'''# S0: STEP S0: CREATION OF SUPPORT CLASSES #'''
'''RATIONALE:   
                We decided to implement this two classes because they will help us to save some data keeping them together in one object, in such a way to not declare every
                time all the variables related to the Candidate Character / Candidate License plate we've found out in the Image.
                - CANDIDATE CHARACTER:      This Class we are going to create will serve us for instantiating Candidate Characters we will find during the program execution
                                            in the image. We define a Candidate Character to be a Contour that must be checked to be a Real Character of the License Plate.
                - CANDIDATE LICENSE PLATE:  This Class we are going to create will serve us for instantiating Candidate License Plate we spot during the Processing of one 
                                            image. We define a Candidate License Plate a set of Characters that are putted together and for which we need to perform some 
                                            checks, in order to find out if it is the Final License Plate.
                - MY ROTATION TRANSFORM &:  The classes below were used to rotate and flip EMNIST Data, making it understable to humans.  :)
                  MY FLIP TRANSFORM         They are not used anymore in the project.
                 
'''

#DEFINING THE CANDIDATE CHARACTER CLASS:
class CandidateCharacter:

    #CONSTRUCTOR DECLARATION:
    def __init__(self, _contour):
        '''CANDIDATE CHARACTER CONSTRUCTOR'''
        self.Bounding_Rectangle = cv2.boundingRect(_contour)                #The method boudingRect draws an approximate rectangle around the Character.
        [x, y, w, h] = self.Bounding_Rectangle                              #Defining x,y coordinate, width and height of the Bouding rectangle.
        self.Bounding_Rectangle_X_Coordinate = x                            #Extracting x coordinate of the Bounding rectangle. 
        self.Bounding_Rectangle_Y_Coordinate = y                            #Extracting y coordinate of the Bounding rectangle.
        self.Bounding_Rectangle_W = w                                       #Extracting width of the Bounding rectangle.
        self.Bounding_Rectangle_H = h                                       #Extracting height of the Bounding rectangle. 
        self.COM_X = (self.Bounding_Rectangle_X_Coordinate +                #X coordinate's Center of Mass. 
                      self.Bounding_Rectangle_X_Coordinate + 
                      self.Bounding_Rectangle_W) / 2     
        self.COM_Y = (self.Bounding_Rectangle_Y_Coordinate +                #Y coordinate's Center of Mass. 
                      self.Bounding_Rectangle_Y_Coordinate + 
                      self.Bounding_Rectangle_H) / 2     
        self.Diagonal = math.sqrt((self.Bounding_Rectangle_W ** 2) +        #Diagonal of the Bounding rectangle. 
                                  (self.Bounding_Rectangle_H ** 2))                                
        self.Ratio_W_H = float(self.Bounding_Rectangle_W /                  #Ratio width/height.
                               self.Bounding_Rectangle_H)                                                   
        self.Bounding_Rectangle_Area = ((self.Bounding_Rectangle_W) *       #Area of the bouding rectangle. 
                                        (self.Bounding_Rectangle_H))                                           
        self.Bounding_Rectangle_Perimeter = ((self.Bounding_Rectangle_W * 2)#Perimeter of the bouding rectangle.
                                        + (self.Bounding_Rectangle_H * 2))                          

    #GETTER METHOD FOR RETURNING THE COM(CENTER OF MASS) OF THE LICENSE PLATE:
    def get_COM(self):
        '''GETTER COM'''
        return (self.COM_X, self.COM_Y)
    
    #GETTER METHOD FOR RETURNING THE DIAGONAL OF THE BOUNDING BOX:
    def get_Diagonal(self):
        '''GETTER DIAGONAL'''
        return self.Diagonal
    
    #GETTER METHOD FOR RETURNING THE AREA OF THE LICENSE PLATE:
    def get_Area(self):
        '''GETTER AREA'''
        return self.Bounding_Rectangle_Area
    
    #GETTER METHOD FOR RETURNING THE AREA OF THE LICENSE PLATE:
    def get_Perimeter(self):
        '''GETTER PERIMETER'''
        return self.Bounding_Rectangle_Perimeter

#DEFINING THE CANDIDATE LICENSE PLATE CLASS:
class CandidateLicensePlate:

    #CONSTRUCTOR DECLARATION:
    def __init__(self):
        '''CANDIDATE LICENSE PLATE CONSTRUCTOR'''
        self.Gray_Img = None
        self.Thresholded_Img = None
        self.License_Plate_Img = None
        self.License_Plate_String = ""
        self.License_Plate_Number_Characters = 0
        self.Rotated_Rectangle_Information = None            #Here we include the plate region center point, width and height and correction angle into rotated rect member 
                                                             #variable of plate.
    
    #GETTER METHOD FOR RETURNING THE LICENSE PLATE STRING:
    def get_License_Plate_String(self):
        '''GETTER LICENSE PLATE STRING'''
        return f"{self.License_Plate_String}"
    
    #GETTER METHOD FOR RETURNING THE LICENSE PLATE IMAGE:
    def get_License_Plate_Img(self):
        '''GETTER LICENSE PLATE IMAGE'''
        return self.License_Plate_Img
    
    #GETTER METHOD FOR RETURNING THE LICENSE PLATE IMAGE THRESHOLDED:
    def get_License_Plate_Thresholded(self):
        '''GETTER LICENSE PLATE IMAGE THRESHOLDED'''
        return self.Thresholded_Img
    
#The classes below were used to rotate and flip EMNIST Data, making it understable to humans  :)
class MyRotationTransform:
    """Rotate by the given angle."""

    #CONSTRUCTOR DECLARATION:
    def _init_(self, angle):
        '''MY ROTATION TRANSFORM CONSTRUCTOR'''
        self.angle = angle    

    def _call_(self, x):
        '''MY ROTATION TRANSFORM CALLER'''
        return TF.rotate(x, self.angle)  #Rotation of the image x by the angle,initialized above.
    
class MyFlipTransform:
    """Rotate by the given angle."""

    #CONSTRUCTOR DECLARATION:
    def _init_(self):
        '''MY FLIP TRANSFORM CONSTRUCTOR'''
        self.self = self

    def _call_(self, x):
        '''MY FLIP TRANSFORM CALLER'''
        return TF.hflip(x)  #Horizontal flip 

'''# S1: STEP S1: CREATION OF SUPPORT FUNCTIONS: PYTHAGOREAN DISTANCE AND TRIGONOMETRY ANGLE #'''
'''RATIONALE:   
                We decided to implement this two Support functions here because they will be used throughout the whole program. In particular, this two classes are: 
                - PYTHAGOREAN DISTANCE: This Function we are going to create will be useful for computing the Distance between two Characters. It corresponds to compute
                                        the Pythagorean Distance. 
                - TRIGONOMETRY ANGLE:   This Function we are going to create will be useful insead for computing the angle between two Characters. We will make use of 
                                        some Trigonometry Tricks in order to compute the desired result, namely the angle in degree between Characters.. 
'''

#DEFINING THE PYTHAGOREAN DISTANCE FUNCTION:
def Pythagorean_Distance(char_candidate, char_comparison):
    #This Function we are going to create will be useful for computing the Distance between two Characters.
    
    return math.sqrt( (( char_candidate.COM_X - char_comparison.COM_X ) ** 2) + ( ( char_candidate.COM_Y - char_comparison.COM_Y ) ** 2) )

#DEFINING THE TRIGONOMETRY ANGLE FUNCTION (SOH CAH TOA):      
def Trigonometry_Angle(char_candidate, char_comparison):
    #This Function we are going to create will be useful instead for computing the angle between two Characters.
    adjecent = float(abs(char_candidate.COM_X - char_comparison.COM_X))
    opposite = float(abs(char_candidate.COM_Y - char_comparison.COM_Y))

    if adjecent != 0.0:                                                 #Check to make sure we do not divide by zero if the center X positions are equal. (float division by 
                                                                        #zero will cause a crash in Python!).
        alpha = (math.atan(opposite / adjecent)) * (180.0 / math.pi)    #If adjacent is not zero, calculate angle.
    else:
        alpha = 1.5708 * (180.0 / math.pi)                              #If adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this 
                                                                        #program.
    return alpha

'''# 00: STEP 0: PRE-PROCESSING OF THE IMAGE #'''
'''RATIONALE:   
                - ORIGINAL IMAGE:   The function, in input receives the Original Image that is taken form the Video. 
                
                - GREYSCALE IMAGE:  From the Original Image, that is in BGR Color Space, we first convert it to the HSV color Space, in such a way to apply our process
                                    directly on it with beacuse it obtains a way better result w.r.t. apply the proces on RGB based images.
                                    Indeed, B, G, R in RGB are all co-related to the color luminance( what we loosely call intensity). This means that we cannot separate 
                                    color information from luminance. HSV("Hue Saturation Value") is used to separate image luminance from color information. This will make
                                    our life easier when we are working on luminance of the image/frame. 
                                    
                - ENHANCING THE IMAGE CONTRASTS: Here we proceed in Enhancing the Image using several of the Morphological Transformations. We have converted images in 
                                    Greyscale exactly for this, since Morphological Transformations are operations that are mostly applied on Binary Images. The reason why 
                                    we decided to go on these Transformations (not done during the course), is because they are really powerful, allowing to get rid of some 
                                    noise, which can disturb the proper processing of our images.
                                    The Morphological Operations used are:
                                    - TOP-HAT TRANSFORMATION: input_img - open_img
                                                              This operation is defined as the difference between the input image and the OPENING Operation. The OPENING 
                                                              operation(Erosion + Dilation) is another Morphological Transformation that performs an EROSION Operation 
                                                              followed by a DILATION Operation. In this way, EROSION will be used to eliminate smaller groups of undesiderable
                                                              pixels (like salt-and-pepper noise), affecting the regions of the image indisciminaely. By performing a DILATION
                                                              ,instead, we will reduce some of these effects.
                                                              (Notice that, for the removal of Salt-and-Pepper Noise, we could have alos used the MEDIAN BLUR studied during the 
                                                              course. However we preferred stay wit the definition).
                                    - BLACK-HAT TRANSFORMATION: input_img - closing_img
                                                              This operation is defined as the difference between the input image and the CLOSING of the input image.  The CLOSING
                                                              operation is a Transformation that is the defined as the opposite of the Openig Transformation. In this case, in 
                                                              fact, the operation performs a Dilation followed by an Erosion. This is commonly used for filling small holes in 
                                                              images.
                                    
                - DE-NOISING THE IMAGE: In this last step of the Pre-Processing we apply the GAUSSIAN BLUR  which is highly effective in removing Gaussian noise from the image.
                                        After this step, we conclude by applying the ADAPTIVE THRESHOLDING, that, w.r.t. the SIMPLE THRESHOLDING (in which for every pixel, the 
                                        same threshold value is applied and therefore if the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to 
                                        a maximum value), if an image has different lighting conditions in different areas, the algorithm determines the threshold for a pixel 
                                        based on a small region(ROIs) around it. So we get different thresholds for different regions(ROIs) of the same image which gives better 
                                        results for images with varying illumination. 
                                        For this function, we need to determine the "blockSize" that characterizes the size of the neighbourhood area and "C", a constant that is 
                                        subtracted from the mean or weighted sum of the neighbourhood pixels. As "adaptiveMethod", that decides how the threshold is computed, we 
                                        will make use of the "cv2.ADAPTIVE_THRESH_GAUSSIAN_C" function in which, the threshold value is a gaussian-weighted sum of the neighbourhood 
                                        values minus the constant C. 
'''

#DEFINE THE PRE-PROCESSING FUNCTION:
def Pre_Process_Img(orig_img):                                                     #Define the function that will be used for Pre-Processing the Original Image.
    
    #PRINTING OUT THE ORIGINAL IMAGE:
    cv2.imshow("Original Image",orig_img)                                         #Print out the Original Image received in Input.
    cv2.waitKey(0)                                                                #Wait for a key to be pressed in order to close the window.
    
    #OBTAINING A GREYSCALE IMAGE:
    h, w, c = orig_img.shape                                                       #Obtaining the height, the width and the number of channels of the image from its shape.    
    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)                            #Converting the Original Image (that is currently in BGR) to the HSV Color Space.
    cv2.imshow("HSV Image",hsv_img)                                               #Print out the HSV-2 Image received in Input.
    cv2.waitKey(0)                                                                #Wait for a key to be pressed in order to close the window.
    _, _, grey_img = cv2.split(hsv_img)                                            #Split the HSV channels obtaining its Hue, its Saturation and its Value channels.
    cv2.imshow("Grayscale Image",grey_img)                                        #Print out the Pre-Processed Greyscale Image received in Input.
    cv2.waitKey(0)                                                                #Wait for a key to be pressed in order to close the window.
    
    #ENHANCING THE IMAGE CONTRASTS:
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))         #Compute the desidered Kernel, passing to it the shape and the size of the kernel.
    top_img = cv2.morphologyEx(grey_img, cv2.MORPH_TOPHAT,                         #Applying the TopHat Transformation.
                                 structuringElement)
    cv2.imshow("Top-Hat Image",top_img)                                           #Print out the To-hat Image received in Input.
    cv2.waitKey(0)                                                                #Wait for a key to be pressed in order to close the window.
    black_img = cv2.morphologyEx(grey_img, cv2.MORPH_BLACKHAT,                     #Applying the BlackHat Transformation.
                                   structuringElement)
    cv2.imshow("Black-Hat Image",black_img)                                       #Print out the Black-Hat Image received in Input.
    cv2.waitKey(0)                                                                #Wait for a key to be pressed in order to close the window.
    grey_img_top = cv2.add(grey_img, top_img)                                      #Add up the Greyscaled Image with the Top-Hat one.
    grey_img_constrast = cv2.subtract(grey_img_top, black_img)                     #Subtract from the Top-Hat Image from the Black-Hat one.
    cv2.imshow("Normal Contrast Image",grey_img_constrast)                        #Print out the Pre-Processed Greyscale Image received in Input.
    cv2.waitKey(0)                                                                #Wait for a key to be pressed in order to close the window.
    
    #DE-NOISING THE IMAGE:
    blur_img = cv2.GaussianBlur(grey_img_constrast, (5, 5), 0)                     #Apply the Gaussian Blur on the Image Enhanced in order to reduce the Noise. 
    cv2.imshow("Gaussian Blurred Image",blur_img)                                  #Print out the Gaussian Blurred Image received.
    cv2.waitKey(0) 
    tresh_img = cv2.adaptiveThreshold(blur_img, 255.0,                             #Apply the adaptive thresholding onto the Gaussian-Blurred image received.
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 19, 9)                #(Originally, we put (19, 9) values here!.
    cv2.imshow("Final Pre-Processed Tresholded Image",tresh_img)                  #Print out the Final Pre-Processed Image received in Input.
    cv2.waitKey(0)                                                                #Wait for a key to be pressed in order to close the window.

    return grey_img, tresh_img                                                     #Returning the final pre-processed image.


'''# 01: STEP 1: DETECT LICENSE PLATES IN THE PREPROCESSED-IMAGE #'''
'''RATIONALE:   
                In this step, after that we have pre-processed the image, we are going to find out the possible License Plate that could be present in the Image we have 
                previously pre-processed (thresholded). 
                Notice that, before the implementation of the "Detect_License_Plates" main function, we will put a Support Function that it will use during its execution. 
                We couldn't put it as an Inner Class because we also use it during the Step 3, when we are finding the Characters present in the Plate. 
                
                We will build up a Hierarchy of Functions, that is:
                
                - DETECT LICENSE PLATES:  This Outer function is the one that will return the List of Candidate License Plates we are going to find out in the Image 
                                          Processed. The Reasoning here is that we first create the List of Candidate License Plates. We then Apply the "Pre_Process_Img"
                                          function declared in step 0, that is performed on the Original Image we pass as argument. We then proceed by finding the List 
                                          of Candidate Characters that we detect in the Image. For doing this we call an Inner Function denoted as "Detect_Characters".
                                          
                                          - DETECT CHARACTERS: This Inner function is the one that has the role to return a list of Candidate Characters in the Image. 
                                                               The reasoning here is to create a List of Characters List. We then compute all the contours present in 
                                                               the image. For each of the Contour we have found, we instantiate a CandidateCharacter object. What we do 
                                                               ,then, is to check whether there is the possibility that this contour can be considered a Candidate 
                                                               Character. For doing this, we make use of an Inner-Inner Function that perform this check. In the end, we 
                                                               return the List of Candidate Characters present in the Image.
                                                               
                                                               - CANDIDATE CHARACTER CHECK: This Inner-Inner function is the one that has the role to check roughly if 
                                                                                            the dimensions of the Candidate Character we've find out is "acceptable". It 
                                                                                            represents a sort of 'First Screening' that is performed on a certain contour to 
                                                                                            see if it could be a character. 
                                                                                            ( Notice that it does not constitute an actual comparison between characters ).
                                                                
                                          Once we have the List of All the Candidate Characters we have found in the Image, we can proceed by finding Candidate License Plates.
                                          In order to do this, we call the Support Function "Character_Matching_M".
                                                                                            
                                          - CHARACTER MATCHING M: This Support function is the one that will return a list of lists(Matrix) of Characters(namely a List of 
                                             (Support Function)   Candidate License Plates) that are matching. We start off with all the possible characters in one big list, 
                                                                  with the aim to re-arrange it into a list of lists of matching chararacters, namely to retur a list of 
                                                                  Candidates License Plates.
                                                                  The reasoning here is to first declare the list of License Plates we are going to return. Then, for each 
                                                                  Candidate Character in the List of all characters that have been detected in the Image, we are going to 
                                                                  find all the characters in the Big List of Detected Characters in the Image that Match the current Character.
                                                                  In the list, of course, add the character itself. For finding this list, we call an Inner-Inner function 
                                                                  "Character_Matching_L" that is able to return the List of Matching Characters.
                                                                  
                                                                  - CHARACTER MATCHING L: This inner function of the Support Function, is the one that has the the purpose of, 
                                                                                          given a Candidate Character and the Big List of Detected Characters in the Image, to 
                                                                                          find all characters in the big list that constitute a match for the single Candidate 
                                                                                          char, and return those matching chars in a list format.
                                                                                          The reasoning here is to, first of all declare the list of Matching Characters we are 
                                                                                          going to return.Then, for every character (that is not, of course the Candidate 
                                                                                          Character) in the Big List of detected Characters, we are going to compute, if it is a
                                                                                          Match. We define a Match to be a Character that is near to another and on the same 
                                                                                          horizontal line. For this reason, we are going to compute the Pythagorean Distance and 
                                                                                          the angle between the two Characters. If they respect certain Threshold that we have 
                                                                                          imposed, we decide to append the Candidate matching Character to the Candidate Plate
                                                                                          List. In the end, we return the Candidate Plate List.
                                                                  
                                                                  We now proceed by checking whether the Candidate Plate List we received is long enough. We choose for this a
                                                                  minimal number of 3 characters that must be present in the Candidate License Plate in order to be considered.
                                                                  If it is ok, we add it to the List of Candidate License Plates. We create then a List containing all the 
                                                                  Candidate License Plates (therefore a Matrix) except for the one we have just added. Note that, here, we also
                                                                  apply the set operation in such a way to consider only a Candidate License Plate once. To continue, we call 
                                                                  recursively the same function, CHARACTER MATCHING M, by passing, as argument, the new List created (the one 
                                                                  without the list we have just added). Then, for each of the Candidate License Plate we have spot during this
                                                                  last recursive step, we add it to the the final List of Candidate Plates, in such a way to provide a final 
                                                                  matrix result.
                                          
                                          Once we have obtained this last result, we can proceed by performing the true Plate Extraction. Namely, we have that this Inner 
                                          function we are going to call, "Plate_Extraction", takes the Candidate License Plate and extract from the Image the Cropped Image of
                                          the License Plate.
                                                                    
                                          - LICENSE PLATE EXTRACTION: This Inner function is the one that has the role to return the Cropped License Plate. Our Rationale here 
                                                                      is to first instantiate an object of the candidate license plate. Consequently, we first sort out the
                                                                      list of characters, depending on their distance. We then compute the COM of the License Plate, its width 
                                                                      and its height. The former is computed by simply doing the subtraction between the outermost and innermost 
                                                                      Character, whilst for the latter we apply the following reasoning. First we sum up all teh height of the
                                                                      characters and we divide it by the number of characters present in the License Plate, in such a way to be 
                                                                      able to compute the Average Height between Characters. Note that to both Height and Width we apply a 
                                                                      multiplication for expanding the Rectangle due to the Padding of the License Plates (because otherwise we 
                                                                      would have a rectangle that is exaclty cropping the characters. To continue, we also compute the correction 
                                                                      angle we have to apply to the plate in order to have a perfectly aligned and straight rectangle(because 
                                                                      otherwise, in some cases, we had the rectangle to be a little bit rotated. We get the Rotation Matrix and
                                                                      we then perform the Rotation Transformation. We can, in the end, return the Cropped Image of the License 
                                                                      Plate.
                                                                
                                          Then, if the Candidate License Plate is Valid, we can add up the Candidate License Plate to the Candidate License Plate List. We can 
                                          then finally return the List of Candidate License Plates.
'''

#DEFINITION OF THE SUPPORT FUNCTION CHARACTER MATCHING M:
def Character_Matching_M(Candidate_Characters_L):
    #This Support function is the one that will return a list of lists(Matrix) of Characters that are matching.
    
    def Character_Matching_L(Candidate_Character, Characters_L):
        #This inner function of the Support Function, is the one that has the the purpose of, given a candidate character and a big list of candidate characters,
        #to find all characters in the big list that are a match for the single possible char, and return those matching chars as a list.
        
        Characters_Matching_L = []                #This will be the return value.

        #CHECKING WHETHER THE CHARACTER WE ARE COMPUTING THE CHARACTERS MATCHING OF, IS NOT THE SAME CHARACTER:
        for Candidate_Matching_Character in Characters_L:                   #For each char (Candidate_Matching_Character) in the big list of Candidate Characters.
            if Candidate_Matching_Character != Candidate_Character:         #If the char we attempting to find matches for is the exact same char as the char in the big list 
                                                                            #we are currently checking, then we should not include it in the list of matches. That would end 
                                                                            #up double including the current char : do not add it to list of matches and jump back to top of 
                                                                            #the for loop.
            
                #COMPUTING DISTANCE AND ANGLE BETWEEN THE TWO CHARACTERS:
                Characters_Distance = Pythagorean_Distance(Candidate_Character, Candidate_Matching_Character)  #Computing the Characters_Distance as the Pythagorean Distance 
                                                                                                               #between the candidate character and the character 
                                                                                                               #(Candidate_Matching_Character) we are currently looking at. 
                Characters_Angle = Trigonometry_Angle(Candidate_Character, Candidate_Matching_Character)       #Computing the Characters_Angle as the Trigonometry_Angle 
                                                                                                               #between the candidate character and the the character 
                                                                                                               #(Candidate_Matching_Character) we are currently looking at. 
                #COMPUTING THE DIFFERENCE IN THE AREA, WIDTH AND HEIGHT OF THE CHARACTERS:
                Area_Difference = (abs(Candidate_Matching_Character.Bounding_Rectangle_Area -                  #Computing the Area_Difference as the difference between 
                                       Candidate_Character.Bounding_Rectangle_Area) /                          #Candidate_Matching_Character (the character we are currently 
                                   float(Candidate_Character.Bounding_Rectangle_Area))                         #looking at) and Candidate Character's Bounding Rectangle Areas, 
                                                                                                               #divided by the Candidate Character's Bounding Rectangle Area.
                        
                W_Difference = (abs(Candidate_Matching_Character.Bounding_Rectangle_W -                        #Computing the Width_Difference as the difference between 
                                    Candidate_Character.Bounding_Rectangle_W) /                                #Candidate_Matching_Character and Candidate Character's 
                                float(Candidate_Character.Bounding_Rectangle_W))                               #Bounding Rectangle Width, divided by the Candidate 
                                                                                                               #Character's Bounding Rectangle Width. 
                                    
                H_Difference = (abs(Candidate_Matching_Character.Bounding_Rectangle_H -                         #Computing the Width_Difference as the difference between 
                                    Candidate_Character.Bounding_Rectangle_H) /                                 #Candidate_Matching_Character and Candidate Character's  
                                float(Candidate_Character.Bounding_Rectangle_H))                                #Bounding Rectangle Height, divided by the Candidate 
                                                                                                                #Character's Bounding Rectangle Height. 
                #CHECKING WHETHER THE DIFFERENCES ABOVE RESPECTS DETERMINATE THRESHOLD(THEY ARE SET EMPIRICALLY):
                if (Characters_Distance < (Candidate_Character.Diagonal * MAX_DIAG_SIZE_MULTIPLE_AWAY) and Characters_Angle < MAX_ANGLE_BETWEEN_CHARS and
                    Area_Difference < MAX_CHANGE_IN_AREA and W_Difference < MAX_CHANGE_IN_WIDTH and H_Difference < MAX_CHANGE_IN_HEIGHT):
                    
                    Characters_Matching_L.append(Candidate_Matching_Character)                                  #If the chars are a match, add the current char to list of 
                                                                                                                #matching chars.
        return Characters_Matching_L                

    
    ##### STARTING OF "Character_Matching_M" FUNCTION (Outer One) #####
    Matching_Characters_M = []                                                  #This will be the return value.

    #FINDING THE LIST OF MATCHING CHARACTERS FOR EVERY CHARACTER:
    for Candidate_Character in Candidate_Characters_L:                          #For each possible char in the one big list of Candidate chars.
        
        #FIND THE MATCHING CHARACTERS:
        Matching_Characters_L = Character_Matching_L(Candidate_Character,       #Find all chars in the big list that match the current char. They are returned as a list.
                                                     Candidate_Characters_L)   
        Matching_Characters_L.append(Candidate_Character)                       #Also add the current char to current possible list of matching chars.
        
        #CHECKING WHETHER WE HAVE AT LEAST 3 CHARACTERS (IN ORDER TO CONSTITUTE A MINIMAL PLATE NUMBER) 
        if len(Matching_Characters_L) >= MIN_NUMBER_OF_MATCHING_CHARS:          #If the current possible list of matching chars is not long enough to constitute a possible 
                                                                                #plate,jump back to the top of the for loop and try again with next char. Note that it's not 
                                                                                #necessary to save the list in any way since it did not have enough chars to be a possible 
                                                                                #plate.If we get here, the current list passed test as a "group" or "cluster" of matching 
                                                                                #chars.
                                                
            #IF WE HAVE AT LEAST 3 CHARACTERS IN THE LIST OF MATCHING, WE ADD THE PLATE TO THE LIST OF PLATES:
            Matching_Characters_M.append(Matching_Characters_L)                 #Add the possible list of matching chars to Matching_Characters_M (the matrix that will 
                                                                                #contain the possible candidate plates).
            
            #CREATING A LIST CONTAINING ALL THE LIST OF MATCHING EXCEPT THE ONE WE'VE JUST FOUND:
            Difference_Matching_Characters_M = list(set(Candidate_Characters_L) #Remove the current list of matching chars from the big list so we don't use those same
                                                    - set(Matching_Characters_L)#chars twice (through the Set operation).Make sure to make a new big list for this since we 
                                                    )                           #don't want to change the original big list. 
            
            #CALLING RECURSIVELY THE FUNCTION ITSELF, IN SUCH A WAY TO FIND ALL THE POSSIBLE MATCHES:
            Recursive_Matching_Characters_M = Character_Matching_M(Difference_Matching_Characters_M)#Recursive call to Character_Matching_M,passing as argument the big 
                                                                                #list above (Difference_Matching_Characters_M).
            for Recursive_Matching_Characters_L in Recursive_Matching_Characters_M:#For each list of matching chars found by the recursive call.
                Matching_Characters_M.append(Recursive_Matching_Characters_L)    #Add it to our original list of lists of matching chars.
                
            #BREAK THE RECURSIVE CALL AND RETURN THE MATCHING CHARACTERS MATRIX CONTAINING ALL THE PLATES:
            break       
    return Matching_Characters_M   #Returning the matrix of the candidate license plates 



#DEFINE THE DETECTION-OF-LICENSE-PLATES FUNCTION:
def Detect_License_Plates(Original_img):
    #This outer function is the one that will return the List of Candidate License Plates we are going to find out in the Image Processed.
    
    def Detect_Characters(Thresholded_img):
        #This inner function is the one that has the role to return a list of Candidate Characters in the Image.
        
        def Candidate_Character_Check(Candidate_Char):
            #This inner-inner function is the one that has the role to check roughly if the dimensions of the character we've find out is "acceptable". 
            
            #PERFORMING THE ROUGH SCREENING:
            #It represents a sort of 'First Screening' that is performed on a certain contour to see if the Candidate_Character could be a character.
            if (Candidate_Char.Bounding_Rectangle_Area > MIN_PIXEL_AREA and Candidate_Char.Bounding_Rectangle_W > MIN_PIXEL_WIDTH and 
                Candidate_Char.Bounding_Rectangle_H > MIN_PIXEL_HEIGHT and MIN_ASPECT_RATIO < Candidate_Char.Ratio_W_H and 
                Candidate_Char.Ratio_W_H < MAX_ASPECT_RATIO):
                return True   #Return True if the candidate character is a "Valid" candidate character. 
            else:
                return False  #Return False otherwise. 

        ##### STARTING OF "Detect_Characters" FUNCTION (Inner One) #####
        
        #DECLARATION OF IMPORTANT VARIABLES:
        Candidate_Characters_L = []                                     #This will be the return value.
        Contours_img, _ = cv2.findContours(Thresholded_img,             #Finding the Contours of the Thresholded image. 
                                           cv2.RETR_LIST, 
                                           cv2.CHAIN_APPROX_SIMPLE)   
        
        #CREATION OF CANDIDATE CHAR & CHECK:
        for i in range(0, len(Contours_img)):                            #For each contour in the image. 
            
            #CREATION OF A CANDIDATE CHARACTER OBJECT:
            Candidate_Character = CandidateCharacter(Contours_img[i])    #Creating a candidate character object. 

            #CHECKING WHETHER IT COULD BE A CANDIDATE CHARACTER:
            if Candidate_Character_Check(Candidate_Character):           #If the contour is a "valid" char...  (note this is not the True comparison!!!(Image to String).
                Candidate_Characters_L.append(Candidate_Character)       #Add it to the list of possible chars.
                
        return Candidate_Characters_L                                    #Return the list of all candidate characters.
    
    
    def Plate_Extraction(Original_img, Matching_Characters_L):
        #This Inner function is the one that has the role to return the Cropped License Plate.
        
        #CREATION OF THE CANDIDATE LICENSE PLATE OBJECT:         
        Candidate_License_Plate = CandidateLicensePlate()                #This will be the return value.
        
        #SORTING THE CHARACTERS OF THE LICENSE PLATE BASING ON THEIR X DISTANCE:
        Matching_Characters_L.sort(key = lambda Character_Matching: 
                                    Character_Matching.COM_X)            #Sort chars from left to right according to X's center of mass.

        #COMPUTING THE CENTER OF MASS (COM) OF THE LICENSE PLATE:        
        License_Plate_COM = (   ( (Matching_Characters_L[0].COM_X + 
                                   Matching_Characters_L[
                                       len(Matching_Characters_L) - 
                                       1].COM_X) / 2.0 ) , 
                                ( (Matching_Characters_L[0].COM_Y + 
                                   Matching_Characters_L[
                                       len(Matching_Characters_L) - 
                                       1].COM_Y) / 2.0) ) 
        

        #COMPUTING THE LICENSE PLATE WIDTH:      
        w_Plate = int((Matching_Characters_L[len(Matching_Characters_L) - 1].Bounding_Rectangle_X_Coordinate + 
                             Matching_Characters_L[len(Matching_Characters_L) - 1].Bounding_Rectangle_W - 
                             Matching_Characters_L[0].Bounding_Rectangle_X_Coordinate) * PLATE_WIDTH_PADDING_FACTOR)

        #COMPUTING THE AVERAGE LICENSE PLATE HEIGHT:
        intPlateHeight = int( (sum([matchingChar.Bounding_Rectangle_H for matchingChar in Matching_Characters_L]) / len(Matching_Characters_L)) * PLATE_HEIGHT_PADDING_FACTOR)

        #COMPUTING THE LICENSE PLATE DISTANCE AND CORRECTION ANGLE:                  
        opposite = Matching_Characters_L[len(Matching_Characters_L) - 1].COM_Y - Matching_Characters_L[0].COM_Y
        pythagorean_distance = Pythagorean_Distance(Matching_Characters_L[0], Matching_Characters_L[len(Matching_Characters_L) - 1])
        alpha_correction = math.asin(opposite / pythagorean_distance) * (180.0 / math.pi)

        #SAVING ALL THE DATA WE HAVE FOUND IN THE OBJECT'S VARIABLE:     
        Candidate_License_Plate.Rotated_Rectangle_Information = ( tuple(License_Plate_COM), (w_Plate, intPlateHeight), alpha_correction )

        #PERFORMING ROTATION TRASFORMATION:
        #GETTING THE ROTATION MATRIX OF THE ALPHA CORRECTION ANGLE: 
        rotationMatrix = cv2.getRotationMatrix2D(tuple(License_Plate_COM), alpha_correction, 1.0)   #It calculates an affine matrix of 2d rotation.
        height, width, _ = Original_img.shape                                                       #Unpacking original image width and height
        imgRotated = cv2.warpAffine(Original_img, rotationMatrix, (width, height))                  #Rotate the entire image
        
        #CROPPING OUT THE LICENSE PLATE NUMBER:
        plate_img = cv2.getRectSubPix(imgRotated, (w_Plate, intPlateHeight),                        #Retrieving the rectangle of the License Plate image. 
                                      tuple(License_Plate_COM))   
        Candidate_License_Plate.License_Plate_Img = plate_img                                       #Copying the cropped plate image into the applicable member variable of 
                                                                                                    #the possible plate.

        return Candidate_License_Plate                                                              #Return the Candidate License Plate. 


    ########## START OF THE "Detect_License_Plates" FUNCTION (Outer One) ##########
    #DECLARATION OF IMPORTANT VARIABLES:
    Candidate_License_Plates_L = []                                                     #This will be the return value.
    h, w, _ = Original_img.shape                                                        #Extracting height,width,number of channels from the original image. 
        
    #IMAGE PRE-PROCESSING:
    _, Thresholded_img = Pre_Process_Img(Original_img)                                  #Preprocessing the image to get grayscale and threshold images.
    
    #FINDING CANDIDATE CHARACTERS IN THE IMAGE:
    Candidate_Characters_L = Detect_Characters(Thresholded_img)                         #Find all possible chars in the scene. This function first finds all contours and 
                                                                                        #then, it only includes contours that could be chars (the comparison to other chars 
                                                                                        #has not been performed yet).
    
    #FINDING THE MATRIX OF MATCHING CHARACTERS: (FIND ALL THE POSSIBLE PLATES) 
    Matching_Characters_M = Character_Matching_M(Candidate_Characters_L)                #Given a list of all possible chars, find groups of matching chars.In the next 
                                                                                        #steps,each group of matching chars will attempt to be recognized as a plate.
    
    #EXTRACTING THE PLATE FOR EACH MATCHING CHARACTERS LIST:
    for Matching_Characters_L in Matching_Characters_M:                                 #For each group of matching chars
        
        #EXTRACTING THE PLATE:
        Candidate_License_Plate = Plate_Extraction(Original_img, Matching_Characters_L) #This is an attempt to extract the license plate.

        #IF THE CANDIDATE PLATE IS VALID:
        if Candidate_License_Plate.License_Plate_Img is not None:                       #Check if the plate was found.
            
            #ADD UP TO THE CANDIDATE LICENSE PLATE LIST THE CANDIDATE LICENSE PLATE
            Candidate_License_Plates_L.append(Candidate_License_Plate)                  #Then,Add it to list of possible plates.
    
    #RETURN THE NUMBER OF CANDIDATE LICENSE PLATES FOUND:
    print(f"\nOur Program has found {str(len(Candidate_License_Plates_L))} Candidate License Plates ")  
    #print(Candidate_License_Plates_L)
    
    return Candidate_License_Plates_L                                                   #Return the list of candidate license plates.


'''# 02: STEP 2: DETECT THE FINAL LICENSE PLATE FROM CANDIDATE LICENSE PLATES #'''
'''RATIONALE:   In this second step, we have the very CORE STEP, in which we come out with the Recognized License Plates List from the Candidate License Plates. Here we 
                construct our Process in a big Function "FINAL LICENSE PLATE EXTRAPOLATION" that embraces other important sub-functions:
                
                - FINAL LICENSE PLATE EXTRAPOLATION:In the outermost function, we first of all check whether the list of Candidate License Plates we get is empty. In the
                                                    case it is not, we start performing some operation on each Candidate License Plate. First and foremost, we start by 
                                                    applying the pre-processing on our cropped candidate license plate image. We then resize(increase) the Binarized image 
                                                    and we continue by applying thresholding in such a way to eliminate gray areas. Then, we find all the Candidate 
                                                    Characters we spot in a Candidate License Plate through the calling to the function "Detect_Characters".
                                           
                                                    - DETECT CHARACTERS: This Inner function is the one that will return the list of all the Candidate Characters that we 
                                                                spot in a Candidate License Plate Image. We first create a copy of the Thresholded Image, and then we find 
                                                                all the contours in the Candidate License Plate Image. Consequenlty, for each of the contour we've found, we 
                                                                opt for the Creation of a CandidateCharacter Object, in such a way to save in a compact way the data  
                                                                related to each Character. We then opt for doing a sort of initial screening in which we discard those 
                                                                Candidate Characters for which we have that they do not respect some minimal length/width and so on and 
                                                                so forth (Note that this check is performed during the Inner-Inner Function "Candidate_Character_Check"). 
                                                                In the end, it returns the list of Candidate Characters.
                                                                - CANDIDATE CHARACTER CHECK: This Inner-Inner function is the one that will handle the screening of the 
                                                                                             Candidate Characters that respects some very rough requirements. It can be
                                                                                             viewed as a sort of "HEURISTIC" in our porgram, since its aim is to weigh 
                                                                                             down the Program in such a way to eliminate, as soon as we find out that, 
                                                                                             some requirements are respected, candidates that in the future would be 
                                                                                             viewed as not valid. We therefore decrease the Program Time complexity of a
                                                                                             lot. 
                                                                                             
                                                    After that, we call the support function "Character_Matching_M", that we have previously introduced suring Step 1, that 
                                                    will return a list of lists(Matrix) of Characters that are matching within the plate. (Have a look at the above Step 1 
                                                    Introduction for a better explanation).
                                                    We then check whether the Matrix we have obtained is currently "Valid", namely if returns at least one list containing the 
                                                    characters of a Candidate License Plate.  
                                                    Then, for every list of Candidate Characters( thus for every Candidate License Plate ), we proceed by sorting out the list, 
                                                    by the distance from the center, in such a way to obtain the Candidate Plate Number to be totally ordered.
                                                    To continue, we perform another set of operations, embraced in the "Handle_Overlapped_Characters", in which, always for every 
                                                    Candidate License Plate, we proceed by removing from them the Overlapping Characters, thus applying a cleaning of the list.
                                           
                                                    - HANDLE OVERLAPPED CHARACTERS: This Inner function is the one that Deletes overlapping Characters from the list of Matching 
                                                                                    Characters. We do this reasoning since we had some problem of "fake contours" we found. Thus, 
                                                                                    we proceed as the following: Whenever we have two characters that are either overlapping or 
                                                                                    are too close to each other to possibly be separate chars, we remove the Smaller Character. 
                                                                                    As we said, we proceed in doing this since we found out that in some cases some contours were 
                                                                                    replicated. As an example, we had that for the letter 'O' both the upper and below part of 
                                                                                    the "ring" may be found as two distinct characters, while for us, we should only include the 
                                                                                    char once.
                                                                                    The reasoning we do is to first create a Deep-Copy of the initial List containing the 
                                                                                    Candidate Characters. We call it "Dirty_List", whilst for the one we return, we assign the 
                                                                                    name "Clean_List" to underline this distinction. We then perform a double loop in which, for
                                                                                    every "Dirty_Character", and for every "Clean_Character" (taking, of course, the two characters 
                                                                                    that are not the same), we check whether, through their Pythagorean Distance, we have that 
                                                                                    they are either Overlapping or it is very very tiny. We take the smaller between the two 
                                                                                    Characters(taking their Area as reference), and we eliminate it from the Clean_List.
                                                                 
                                                    We proceed by taking the Index of the list that has the most characters and we assume, by taking a Greedy Choice, that it 
                                                    corresponds to the actual list of Characters. After having choosen the list, we perform the true Recognition, through which 
                                                    we are able to recognize the actual characters present in the License Plate Image. Note that we are still in the loop, thus, 
                                                    we will have to perform this operation multiple times. 
                                                    - RECOGNITION LICENSE PLATE: This Inner function is the one that will perform the actual Recognition of the Characters in 
													                             the Plate Image. We first convert back the thresholded image(binarized), and therefore in 
                                                                                 Gray, to BGR color Space. Then, for each character in the License Plate, we crop out the 
                                                                                 character, creating a sort of ROI(Region of interest). We apply the Resizing Transformation. 
                                                                                 We flatten the image in 1D, converting it into float. At this point, we have implemented two 
                                                                                 procedures to recognize the license plate : RECOGNITION LICENSE PLATE KNN and RECOGNITION 
                                                                                 LICENSE PLATE CNN.
                                                                                 However,The KNN approach is not used anymore in our program,as it is not able to recognize 
                                                                                 the entire license plate.KNN is a very strict algorithm : it needs in input very similar 
                                                                                 samples images w.r.t. to the Images on which it has been traned, otherwise, it will miserably 
                                                                                 fail (as in our case unfortunately). 
                                                                                 On the other hand, The CNN approach gives us the desired result. As Recognized Character, it 
                                                                                 returns a list of the probabilities for each of the 36 classes we have.
                                                                                 Then, we do the argmax on the index with higher probability, in order to take the more probable 
                                                                                 class that corresponds to the CNN Prediction. The Recognized Character is added to the string 
                                                                                 that, at the end, will contain the entire License Plate.
                                                    At the end of the outer function, we will sort the list of Candidate License Plates, returning as "Final_License_Plate" the 
                                                    one that has the greatest number of recognized characters in the sorted list.
'''

def Final_License_Plate_Extrapolation(Candidate_Plates_List, Model):
    #This Outer function is the one that will return the Recognized License Plate 
    
    ###################################################################################################
    def Detect_Characters(Thresholded_Image):
        #This Inner function is the one that will return the list of all the Candidate Characters that we spot in a Candidate License Plate Image.
        
        def Candidate_Character_Check(Candidate_Character):
            #This Inner-Inner function is the one that handles a sort of screening in checking whether the Candidate Character could be really a Character.
            
            #This function is a 'first pass': it does a rough check on a contour to see if it could be a char. Again, Note that we are not (yet) comparing the char to other 
            #chars to look for a group.
            if (Candidate_Character.Bounding_Rectangle_Area > MIN_PIXEL_AREA and Candidate_Character.Bounding_Rectangle_W > MIN_PIXEL_WIDTH and 
                Candidate_Character.Bounding_Rectangle_H > MIN_PIXEL_HEIGHT and MIN_ASPECT_RATIO < Candidate_Character.Ratio_W_H and Candidate_Character.Ratio_W_H < MAX_ASPECT_RATIO):
                return True        #Return True if it is a valid char. 
            else:
                return False       #Return False otherwise. 
 
        ##### START OF THE "findPossibleCharsInPlate" FUNCTION (Inner One) #####
        
        #FINDING ALL THE CONTOURS IN THE CANDIDATE LICENSE PLATE IMAGE:     
        Countours_Img, _ = cv2.findContours(Thresholded_Image,              #Finding all the contours in the license plate.
                                            cv2.RETR_LIST, 
                                            cv2.CHAIN_APPROX_SIMPLE)   

        #BUILD UP THE CANDIDATE CHARACTERS LIST:
        Candidate_Characters_L = []                                         #This will be the return value
        for Contour in Countours_Img:                                       #For each contour.
            
            #CREATION OF A CANDIDATE CHARACTER OBJECT:
            Candidate_Character = CandidateCharacter(Contour)               #Creating a candidate character object. 

            #CHECKING WHETHER IT IS A "VALUABLE" CANDIDATE CHARACTER:
            if Candidate_Character_Check(Candidate_Character):              #If the contour is a "Valid" char.
                Candidate_Characters_L.append(Candidate_Character)          #Add it to the list of possible chars.
     
        return Candidate_Characters_L                                       #Return the candidate character list 


    #If some characters are overlapping or too close to each other to separate and distinguish them, remove the inner (smaller) char. This is to prevent including the same 
    #char twice,if two contours are found for the same char.
    #Example :For the letter 'O',both the inner and the outer rings may be found as contours, but we should only include the char once!! We introduced this function for 
    #solving this redundancy problem.
    def Handle_Overlapped_Characters(Dirty_List):
        #This Inner function is the one that Deletes overlapping Characters from the list of Matching Characters.
        
        #CREATION OF A DEEP-COPY OF THE "DIRTY" MATCHING CHARACTERS LIST.
        Clean_List = list(Dirty_List)                                       #This will be the return value.

        #REMOVING FROM THE DIRTY LIST THE OVERLAPPING CHARACTERS:
        for Dirty_Character in Dirty_List:                                  #For each Dirty_character in the list of matching characters (Dirty_List).
            for Clean_Character in Dirty_List:                              #For each Clean_Character in the list of matching characters(Dirty_List).
                if Dirty_Character != Clean_Character:                      #If the current char is different from the other char.  
                    
                    #CHECKING WHETHER THE PYTHAGOREAN DISTANCE BETWEEN CHARACTERS IS VERY TINY:
                    if Pythagorean_Distance(Dirty_Character, Clean_Character) < (Dirty_Character.Diagonal * MIN_DIAG_SIZE_MULTIPLE_AWAY):   
                        #!!We have found overlapping chars!! if that char was not already removed on a previous pass, remove it.
                        
                        #CHECKING WHICH CHARACTER IS SMALLER AND ELIMINATE IT FROM CLEAN_LIST:
                        if Dirty_Character.Bounding_Rectangle_Area < Clean_Character.Bounding_Rectangle_Area:   #If the current char is smaller than the other char.
                            if Dirty_Character in Clean_List:               #If the current char was not already removed in the previous "pass".
                                Clean_List.remove(Dirty_Character)          #Then, Remove the current char.
                        else:                                               #Otherwise,if the other char is smaller than the current char.
                            if Clean_Character in Clean_List:               #If the other char was not already removed in the previous "pass".
                                Clean_List.remove(Clean_Character)          #Then,Remove other char.
        return Clean_List                                                   #Return Clean_List, the list without overlapping characters.


    #NOTE HERE: 
    #Recognition_License_Plate_KNN is not used anymore. A CNN has been used to recognize the Characters in the Plate Image (later on..).
    #We implemented it as first way, for approaching our problem, but it was a very bad idea :) .We have decided to leave the KNN recognition code just for curiosity. 
    #It recognizes some characters, but it is not able to recognize the entire license plate almost never. This is due to the fact that, since it is a very strict ML
    #algotithm, it needs in input very similar samples images w.r.t. to the Images on which it has been traned, otherwise, it will miserably fail (as in our case unfortunately). 
    
    def Recognition_License_Plate_KNN(Thresholded_Image, Matching_Characters_L, Model):
        #This Inner function is the one that will perform the actual Recognition of the Characters in the Plate Image using KNN. 
        
        #DECLARING SOME USEFUL VARIABLES:
        Final_License_Plate = ""                                                #This will be the return value, the chars in the license plate.
        # cv2.imshow("Final Threshlded License Plate", Thresholded_Image)       #Re-show scene image.
        # cv2.waitKey(0)   
        
        #FOR EACH CHARACTER IN THE SELECTED LICENSE PLATE:
        for Matching_Character in Matching_Characters_L:                        #For each char in the license plate.
            
            #CREATION OF A ROI(REGION OF INTEREST) SMALLER IMAGE THAT CROPS THE CHAR OUT OF THE SELECTED LICENSE PLATE IMAGE.
            ROI_Character_Img = Thresholded_Image[Matching_Character.Bounding_Rectangle_Y_Coordinate : Matching_Character.Bounding_Rectangle_Y_Coordinate + Matching_Character.Bounding_Rectangle_H,
                                       Matching_Character.Bounding_Rectangle_X_Coordinate : Matching_Character.Bounding_Rectangle_X_Coordinate + Matching_Character.Bounding_Rectangle_W]
            
            ROI_Character_Img= cv2.copyMakeBorder(ROI_Character_Img, Matching_Character.Bounding_Rectangle_H//5,    #Adding extra padding to the image.
                                                  Matching_Character.Bounding_Rectangle_H//5, Matching_Character.Bounding_Rectangle_W//2, 
                                                  Matching_Character.Bounding_Rectangle_W//2, cv2.BORDER_CONSTANT,(0,0,0))   
            
            #APPLYING RESIZING TRANSFORMATION:
            #print(ROI_Character_Img.shape)
            # cv2.imshow("ROI", ROI_Character_Img)                                   #Re-show scene image
            # cv2.waitKey(0)  
            
            ROI_Character_Img = cv2.resize(ROI_Character_Img,                        #Resize the image. It is necessary for char recognition.
                                           (RESIZED_CHAR_IMAGE_WIDTH, 
                                            RESIZED_CHAR_IMAGE_HEIGHT))           
            #print(ROI_Character_Img.shape())
            cv2.imshow("License Plate Character", ROI_Character_Img)                 #Re-show scene image.
            cv2.waitKey(0)
        
            #FLATTENING THE IMAGE IN A 1D ARRAY & CONVERTS TO FLOAT NUMBERS.
            ROI_Character_Flattened_Img = np.float32(ROI_Character_Img.reshape((1,   #Flatten image into 1d numpy array.
                                                    RESIZED_CHAR_IMAGE_WIDTH * 
                                                    RESIZED_CHAR_IMAGE_HEIGHT)))       
            
            #APPLYING THE K-NEAREST ALGORITHM (IN THIS CASE), AND GETTING THE RECOGNIZED CHAR
            Recognized_Character = Model.predict(ROI_Character_Flattened_Img)        #Predict the labels of the data values on the basis of the trained model.
            
            #Recognized_Character = str(chr(int(Recognized_Character[0])))           #Get character from results
            Recognized_Character = dict_labels_to_alpha[Recognized_Character[0]]     #Handling Labels-Alphanumeric recognitions through dictionary definitions
            print('Recognized_Character', Recognized_Character)
            Final_License_Plate = Final_License_Plate + Recognized_Character         #Append the current char to the full string

        return Final_License_Plate                                                   #Return the recognized License Plate 

    #NOTE HERE: This is the function we really use!
    #Recognition_License_Plate_CNN : We are able to recognize the characters of the License plate using CNN.
    def Recognition_License_Plate_CNN(Thresholded_Image, Matching_Characters_L, Model):
        #This Inner function is the one that will perform the actual Recognition of the Characters in the Plate Image. 
        
        #DECLARING SOME USEFUL VARIABLES:
        Final_License_Plate = ""                                                    #This will be the return value,the characters in the license plate 
        
        #FOR EACH CHARACTER IN THE SELECTED LICENSE PLATE:
        for Matching_Character in Matching_Characters_L:                            #For each char in the license plate
            
            #CREATION OF A ROI(REGION OF INTEREST) SMALLER IMAGE THAT CROPS THE CHAR OUT OF THE SELECTED LICENSE PLATE IMAGE.
            ROI_Character_Img = Thresholded_Image[Matching_Character.Bounding_Rectangle_Y_Coordinate : 
                Matching_Character.Bounding_Rectangle_Y_Coordinate + 
                Matching_Character.Bounding_Rectangle_H, 
                Matching_Character.Bounding_Rectangle_X_Coordinate : 
                    Matching_Character.Bounding_Rectangle_X_Coordinate + 
                    Matching_Character.Bounding_Rectangle_W]
            
            ROI_Character_Img= cv2.copyMakeBorder(ROI_Character_Img,                #Adding extra padding to the image.
                            Matching_Character.Bounding_Rectangle_H//5, 
                            Matching_Character.Bounding_Rectangle_H//5, 
                            Matching_Character.Bounding_Rectangle_W//2, 
                            Matching_Character.Bounding_Rectangle_W//2, 
                            cv2.BORDER_CONSTANT,(0,0,0))   
            #APPLYING RESIZING TRANSFORMATION:
            #print(ROI_Character_Img.shape)
            # cv2.imshow("ROI", ROI_Character_Img)                                  #Re-show scene image.
            # cv2.waitKey(0)  
            
            ROI_Character_Img = cv2.resize(ROI_Character_Img,                       #Resize the image.It is necessary for character recognition.
                                           (RESIZED_CHAR_IMAGE_WIDTH, 
                                            RESIZED_CHAR_IMAGE_HEIGHT))    
            #print(ROI_Character_Img.shape())
            cv2.imshow("License Plate Character", ROI_Character_Img)                #Re-show scene image.
            cv2.waitKey(0)
            
            
            #FLATTENING THE IMAGE IN A 1D ARRAY & CONVERTS TO FLOAT NUMBERS.
            ROI_Character_Flattened_Img = np.float32(ROI_Character_Img.reshape(     #Flatten image into 1d numpy array.
                (1,1, RESIZED_CHAR_IMAGE_WIDTH , RESIZED_CHAR_IMAGE_HEIGHT)))        
            
            Recognized_Character = Model(torch.tensor(ROI_Character_Flattened_Img)) #It returns a list of the probabilities for each of the 36 classes. 
            Recognized_Character=np.argmax(Recognized_Character.tolist())           #Doing the argmax on the index with higher probability, in such a way to take the 
                                                                                    #more probable class that corresponds to the CNN Prediction.
            
            #Recognized_Character = str(chr(int(Recognized_Character[0])))          #Get character from results.
            Recognized_Character = dict_labels_to_alpha[Recognized_Character]       #Handling Labels-Alphanumeric recognitions through dictionary definitions.
            print('Recognized_Character', Recognized_Character) 
            Final_License_Plate = Final_License_Plate + Recognized_Character        #Append current char to full string.

        return Final_License_Plate                                                  #Return a candidate license plate.


    ########## START OF THE "Final_License_Plate_Extrapolation" FUNCTION (Outer One) ##########
    #CHECKING WHETHER THE LIST OF CANDIDATE PLATES IS EMPTY:
    if len(Candidate_Plates_List) == 0:                                             #If the list of possible plates is empty.
        print('There is no Candidate License Plate. I am sorry...')
        return Candidate_Plates_List                                                #Return the list of candidate plates.
    #At this point,we can be sure the list of possible plates has at least one plate.
        
    #PERFORMING A SEQUENCE OF OPERATION ON EACH CANDIDATE PLATE:
    for Candidate_Plate in Candidate_Plates_List:                                   #For each possible plate, this is a big for loop that takes up most of the function.

        #APPLYING THE PRE-PROCESSING ON OUR CROPPED CANDIDATE LICENSE PLATE IMAGE:
        Candidate_Plate.Gray_Img, Candidate_Plate.Thresholded_Img = Pre_Process_Img(Candidate_Plate.License_Plate_Img)#Preprocessing the image to get grayscale and threshold images.
        
        #APPLYING THE RESIZING TRANSFORMATION:       
        Candidate_Plate.Thresholded_Img = cv2.resize(Candidate_Plate.Thresholded_Img,#Increase the size of the plate image for easier viewing and char detection.
                                                     (0, 0), fx = 2, fy = 2)     

        #APPLYING THRESHOLDING IN ORDER TO ELIMINATE ANY GRAY AREAS:                
        _, Candidate_Plate.Thresholded_Img = cv2.threshold(Candidate_Plate.Thresholded_Img, #Applying threshold again to eliminate any gray areas.
                                                           0.0, 255.0, 
                                                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)    
        
        #FINDING ALL THE CANDIDATE CHARACTERS IN THE CANDIDATE PLATE:        
        Candidate_Characters_L = Detect_Characters(Candidate_Plate.Thresholded_Img)   #Find all possible chars in the plate.This function first finds all contours, then only 
                                                                                      #includes contours that could be chars (without comparison to other chars yet).
                
        #CALLING THE SUPPORT FUNCTION "Character_Matching_M": 
        Matching_Characters_M = Character_Matching_M(Candidate_Characters_L)          #Given a list of all possible chars, find groups of matching chars within the plate.

        #CHECKING WHETHER THE LIST OF LISTS OBTAINED IS "VALID" (NOT EMPTY):
        if (len(Matching_Characters_M) != 0):                                         #If the groups of matching chars were found in the plate.
                
            #FOR EVERY INDEX IN THE RANGE OF THE LENGTH OF THE MATRIX MATCHING CHARACTERS, SORT & REMOVE OVERLAPPING CHARACTERS:
            for i in range(0, len(Matching_Characters_M)):                            #For each list of matching chars.
                
                #SORTING OUT THE LIST DEPENDING ON THE DISTANCE FROM THE CENTER:
                Matching_Characters_M[i].sort(key = lambda matchingChar:              #Sort charcacters from left to right according to X's center of mass.
                    matchingChar.COM_X)        
                
                #REMOVE INNER OVERLAPPING CHARACTERS(CLEAN OUT THE MATRIX FROM OVERLAPPING CHARACTERS):
                Matching_Characters_M[i] = Handle_Overlapped_Characters(Matching_Characters_M[i])#Remove inner overlapping characters within each possible plate.
            
            # "HEURISTIC" : We assume that the longest list of potential matching characters is the actual list of chars (the license plate)
            
            #TAKING THE INDEX OF THE LIST THAT HAS THE HIGHEST NUMBER OF CHARACTERS:
            Index_Long = max(enumerate(Matching_Characters_M),                        #Taking the index of the list that has the highest number of characters,according to 
                             key=lambda sub: len(sub[1]))[0]                          #the heuristic.

            #MAKING A GREEDY CHOICE SUPPOSING THAT THE "RIGHT" LIST OF CHARACTERS IS THE LONGEST ONE:   
            List_Long = Matching_Characters_M[Index_Long]                             #Taking the List with the highest number of characters, according to the heuristic.
            #Candidate_Plate.License_Plate_String = Recognition_License_Plate_KNN(Candidate_Plate.Thresholded_Img, List_Long, Model)      #Return the License Plate always in Upper-case, in such a way to also guarantee that there could be the case for which p and P are missclassified, but the real license plate is always been returned
            
            Candidate_Plate.License_Plate_String = Recognition_License_Plate_CNN(Candidate_Plate.Thresholded_Img, #Return the License Plate always in Upper-case, 
                                                                                 List_Long,                       #in such a way to also guarantee that there could be the case 
                                                                                 Model)                           #for which p and P are missclassified, but the real license 
                                                                                                                  #plate is always been returned.
            Candidate_Plate.License_Plate_Number_Characters =  len(str(Candidate_Plate.License_Plate_String))     #Lenght of the license plate string.
            #print(f"We have Recognized a new License Plate: {Candidate_Plate.License_Plate_String} composed of {Candidate_Plate.License_Plate_Number_Characters} characters")

    #SORTING OUT THE LIST OF CANDIDATE PLATES BY LENGTH:
    Candidate_Plates_List.sort(key = lambda possiblePlate: len(possiblePlate.License_Plate_String), reverse = True)#Sorting the list of candidate plates by length.
    
    #EXTRACT THE FINAL LICENSE PLATE, SUPPOSING THAT IT IS THE ONE WITH THE HIGHEST NUMBER OF RECOGNIZED CHARACTERS:
    Final_License_Plate = Candidate_Plates_List[0]                                  #Extracting the final license plate, that is the one with the highest number of 
                                                                                    #recognized characters. 
    
    return Final_License_Plate   #Return the Recognized License Plate


'''# 03: STEP 3:WRAPPING UP ALL THE PIECES: RETRIEVE LICENSE PLATE, DRAW IT ON VIDEO AND RECOGNIZE THE LICENSE PLATE #'''
'''RATIONALE:   In this wrapping step, we are goig to collect all the pieces scattered throughout the Program. It works as a sort of Main Function.
		        Before describing the relevant functions of this step, we define some useful support functions and classes:
		    
		        #SUPPORT FUNCTIONS: 

			    -PLOTTING DATASET : It is used in the inner function Loading_and_Download_Dataset_EMNIST (an inner function of Detect_and_Recognize_License_Plate (KNN)). 
						            Also this latter function is not used anymore - we have left it just for curiosity. 
						            We were trying to use the EMNIST dataset, but, at the end, we have opted for creating one on our own.     

			    -LOADING AND DOWNLOADING DATASET : This support function is the one that will undergo a search process on the OS Folder in such a way to map every folder
                                    that corresponds, in our case, to the label of the files, to the Sample Images. We will proceed in this way, by creating a csv where
                                    we are going to save the correspondence label-image. We then proceed to return, at the end, the tensors corresponding of list labels 
                                    and list of the images.


			    -GET VIDEO : It is used to take the license plate opening the camera of our PC.

			    -GET FROM VIDEO : It is used to take the license plate directly from a video. 

			    -GET LEVEL OF BRIGHTNESS: It returns the level and the color for both the area of the General Description and of the License Plate w.r.t. its level of 
                                          brightness. We suppose the License Plate Description will occur in the second horizontal half of the image. At least for us,
                                          this is a very coherent supposition since this application will be used for Detecting and Recognizing a License Plate at the 
                                          entrance of an hypothetical company X. We take the image from the input, and we are going to take these two cropped zones of 
                                          the image. We first instantiate a sort of Level Dictionary :we assign and establish the Level Value and the Level Color that 
                                          will correspond to a certain range. Then, in order to get the Average Brightness of the Image, we convert them to the HSV 
                                          Color Space, in such a way to get the "Value" channel, representing the Brightness. We compute the average Brightness and then
                                          we check whether the values obtained are in the range of one of the levels pre-established. Once we find the range in which 
                                          they fall, for both the two images, we save their level description, their color level description, their color name description.  

                #SUPPORT CLASSES: 
                
		        - MYDATASET : This class MyDataset has been created to take the labels.
                			  It returns the image with the corresponding label (The label is the name of the folder that contains the image)

                -CUSTOM DATASET : The dataset has been extracted by hand since labels correspond to the name of the folder

                -LICENSE PLATE CNN : In this class, we define our convolutional neural network. 
                				 In the init function, we define the fully connected layers in our neural network,mainiting consistency between the 
                				 output of a layer and the input of the next layer
                				 The forward function will pass the data through all the layers. 

		    Back to the relevant functions.. 
			
		    -DETECT AND RECOGNIZE LICENSE PLATE_CNN : This function embraces the whole program.First, It gets data and labels by calling Loading_and_Downloading_Dataset, 
                                                      a support function described above. It defines the dataframe and makes it a dataset thanks to Custom_Dataset class.
                                                      At the end, it will return the recognized license plate. 
                                                      It is made of several subfunctions:
                                                      -SAMPLE FROM CLASS : It performs a train-test manual split.Once we have obtained the train and test's data and labels, 
                                                                           it returns a tuple of two datasets: The TrainDataset and TestDataset will contain,respectively,
                                                                           the train data and label and the test data and label. 
														   
													  -TRAIN AND TEST : It has been created just for code compactness.Once it is called, both the training and the test loop 
                                                                        (inner functions) will be executed.
																		-TRAINING LOOP: It will print the value of the loss. 
																	    -TEST LOOP : It will print the value of the accuracy with the average loss. 
                                                                        We extract the train_dataset and the test_dataset calling the 'manual split' function. We get the 
                                                                        Train_Dataloader and the Test_Dataloader. We instatiate our model and define the loss function and 
                                                                        the optimizer. 
     																	At this point, we have decided to load the model.This acts as a sort of File Tuning, that downloads 
                                                                        the weights of precedent Models we have downloaded, allowing us to start doing the training and 
                                                                        testing from a better accuracy. 
       																    Thanks to it, we were able to reach an accuracy near to 96%. It was a big improvement, but not all
                                                                        the license plates were recognized correctly.Reaching a higher accuracy may cause overfitting problems.
      																    For each epoch, we call the train and test loop and save the model.
                                                                        We can continue with the Detection of the List of Candidate License plate, calling the function 
                                                                        "Detect_License_Plates". We get a list of Candidate License Plate : if it is not empty, we can 
                                                                        "figure out" the Final License Plate, namely the true string containing the True License Plate. 
                                                                        Once we have the final string,we draw a Rectangle that marks the Location of the License Plate. We 
                                                                        compute the current date and time and we add those information on the Frame, with the Id of the 
                                                                        detected Employee and its License Plate.We will get the information of the Employee from the database. 
                                                                        Near the Actual License Plate, the Recognized License Plate will be printed. We get the size and the 
                                                                        center of mass of the plate to print it well. In the end, we save the Frame.
-----LAST BUT NOT LEAST..

Notice that we have tried to recognize the license plate using KNN too.However, this does not perform well in most of the cases.
                - DETECT AND RECOGNIZE LICENSE PLATE  (KNN): This is the function that embraces the whole program. First and foremost, it checks whether the Loading of Data 
                                                             and the Training using KNN Algorithm (or other methodology), has succeded. Here, we call one function, that is 
                                                             one of the inner funciton of this Outer Function, that has the role to do the things described above.
                
                                                             - LOADING & TRAINING KNN: This Inner Function is the one that has the role to load the Data we are going to use 
                                                                                       during the KNN Classification. The reasoning here is to first Load the Classification 
                                                                                       and the Flattened Images. In the meanwhile, we also Reshape the Classifier and we let 
                                                                                       it to be in 1D in order to pass it to the Training Step. We then set the number of 
                                                                                       Neighbours to 1, and we, finally, Train the Classifier. 
                                                                                       We then proceed in the Loading of the Video/Frame through which we are going to Detect 
                                                                                       and Recognize the License Plate. To continue, we compute the Brightness Level of the 
                                                                                       Image. In particular, here we call another Inner function, that will return us the 
                                                                                       brightness level and color for both the first 1/8 of the image and for the plate area.
                                                    
                                                                        
                                                             We can continue with the Detection of the List of Candidate License plate, calling the function 
                                                             "Detect_License_Plates". We get a list of Candidate License Plate : if it is not empty, we can "figure out" the
                                                             Final License Plate, namely the true string containing the True License Plate. Once we have the final string,we 
                                                             draw a Rectangle that marks the Location of the License Plate. We compute the current date and time and we add 
                                                             those information on the Frame,with the Id of the detected Employee and its License Plate.We will get the 
                                                             information of the Employee from the database. Near the Actual License Plate, the Recognized 
									                         License Plate will be printed.We get the size and the center of mass of the plate to print it well. In the end,
                                                             we save the Frame. it out in the right manner. In the end, we save the Frame.

'''

#NOTICE : Plotting_Dataset is not used anymore in our program. It is used in the inner function Loading_and_Download_Dataset_EMNIST (later on..). Also this latter function 
#is not used anymore - we have left it just for curiosity. We were trying to use the EMNIST dataset, but, at the end, we have opted for creating one on our own.    
def Plotting_Dataset(dataset):    
    figure = plt.figure(figsize = (10,10))                  #Creating a 10x10 figure.
    cols, rows = 3,3                                        #Setting the number of columns and rows. 
    for i in range(1, cols*rows+1):
        img, label = dataset[i]                             #Getting the image and label. 
        img = TF.rotate(img, 90)                            #Rotate the image by 90 degree.
        figure.add_subplot(rows, cols, i)           
        plt.imshow(img.squeeze(), cmap = 'gray')            #Processing the squeezed image and displaying its format.
        plt.xlabel(label, color = 'red', loc = 'center')
    plt.show()                                              #Showing the image. 
    return
    
#We construct our own Dataset. We need to define our own class that must necessarily expand the init, len and getitem class.
class MyDataset(Dataset):
    def __init__(self, images, labels, images_path, transform = None):   
        #Labels are stored in a csv file, each line has the form namefile,label - Ex:img1.png, dog; transforms is a list of transformations we need to apply to the data.
        self.labels = labels                                            #Initializing the labels.
        self.images = images                                            #Initializing the images.
        self.transform = transform                                      #Initializing the transformations. 
        self.images_path = images_path                                  #Initializing the images_path. 
        
    def __len__(self):
        return len(self.labels)                                         #Return the length of the labels. 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):                                        #Transofrm the idx to list if it is a tensor.
            idx = idx.tolist()             
            
        #Reading the Image.
        #print(self.images_path)
        img_path = os.path.join(self.images_path, self.labels[idx],     #Defining the image path.
                                self.images[idx])      
        #print(img_path)
        #print('The image path is: ', img_path)
        #img = read_image(img_path).view(1, -1)
        img = read_image(img_path)                                      #Reading the image. 
        img = np.array(img)                                             #Converting the image into a numpy array. 
        #print("The shape of the Read Image is: ", img.shape)

        #Getting the label.
        label = self.labels[idx]                                        #Taking the label.

        #Apply the Transformations.
        if self.transform:                                              #If the transform will contain a list of transformations,
            img = self.transform(img)                                   #All the transformations will be applied to the image.
            
        img = torch.reshape(img, (1, 28, 28))                           #The image is returned as a tensor with shape 1,28,28.
        return img, label                                               #Return the image with the corresponding label.

def Loading_and_Download_Dataset():

    dir_path = 'Alpha_Numeric_Customized_Dataset/Alpha_Numeric_Customized_Dataset'     #Defining the folder path. 
    #List to store files name
    list_files = []                                                     #Initializing the list of files. 
    list_labels= []                                                     #Initializing the list of labels. 
    for (dir_path, dir_names, file_names) in walk(dir_path):    
        for filename in file_names:                                     #For each filename in file_names. 
            if filename.endswith(".jpg"):                               #If the file extension is 'jpg'.
                list_files.append(filename)                             #Append the filename to the list of files. 
                list_labels.append(dir_path[-1])                
                
    with open('Alpha_Numeric_Customized_Dataset/Alpha_Numeric_Customized_Dataset.csv',  #Opening 'Dataset_Alphabet_Numbers/Alpha_Numeric_Customized_Dataset.csv' in writing mode.
              'w', encoding = 'utf-8', newline='') as outfile:     
        rowlists = zip(list_files, list_labels)
        writer = csv.writer(outfile)
        writer.writerow(('Image', 'Label'))
        for row in rowlists:
            writer.writerow(row)
        mydataset = MyDataset(images=list_files, labels=list_labels, images_path="Alpha_Numeric_Customized_Dataset/Alpha_Numeric_Customized_Dataset", transform=ToTensor())

        '''
        fig = plt.figure()
        figure = plt.figure(figsize = (8,8))
        cols, rows = 3,3
        for i in range(1, cols*rows+1):
            img, label = mydataset.__getitem__(i)
            img=TF.rotate(img,270)
            img=TF.hflip(img)
            figure.add_subplot(rows, cols, i)
            plt.imshow(img.squeeze(), cmap = 'gray')
        plt.show()
        '''
                
        list_labels = [dict_alpha_to_labels[ch] for ch in list_labels]  
        list_labels = torch.tensor(list_labels)                         #The list of labels is converted into a tensor. 
                
        list_images=[]                                                  #Initializing the list of images. 
        for i in range(len(list_files)): 
            img,_=mydataset.__getitem__(i)   
            img=TF.rotate(img,270)                                      #Rotate the image by 270 degree.  
            img=TF.hflip(img)                                           #Horizontal flip the image. 
            img.view(1,784)                                             #The image is converted into a tensor of shape (1,784)  (notice that the image's shape is 28x28).
            img=img.tolist()                                            #Converting the image into a list. 
            list_images.append(img)                                     #Append the image to the list of images. 
        list_images= torch.tensor(list_images).view(36576,784)          #The list of images is converted into a tensor of shape (36576,784).
            
        return list_labels,list_images                                  #Return the list of labels and the list of images. 
            
            
            
def get_Video(dt):
    Video = cv2.VideoCapture(0)                                         #Getting a video capture object for the camera.
    while True:
        #We read a frame from the video stream.
        Image = Video.read()[1]
        cv2.imshow("Live Video", Image)                                 #Showing the Image. 
        k = cv2.waitKey(1)                                              #Wait for a key to be pressed in order to close the window.
        if k == ord('q') or k == ord(' ') or k == ord('r') or k == ord('s'):#Pressing q on the keyboard will return the Image. 
            return Image

def get_from_Video(Saved_Video):
    #Load the color image.
    video = cv2.VideoCapture(Saved_Video)                               #Getting a video capture object for the camera. 
    
    while True:
        #We read a frame from the video stream.
        Image = video.read()[1]
        
        cv2.imshow("Pre-Registered Video", Image)                       #Showing the Image. 
        k = cv2.waitKey(1)                                              #Wait for a key to be pressed in order to close the window.
        if k == ord('q') or k == ord(' ') or k == ord('r') or k == ord('s'):
            return Image
    
def Get_Level_Brightness_Image(img):
    #This Inner function has the role to return a color (for both the areas of the General Description and the License Plate ) w.r.t. its level of brightness. 
    #It will be used for finding different colors for the text to put in the image.

    #TAKING THE SHAPE OF THE IMAGE:
    h, w, c = img.shape                     #Getting the height, width, number of channels of the image 

    #LISTING ALL POSSIBLE LEVELS OF BRIGHTNESS WITH THEIR COLOR, DEPENDING ON THEIR VALUE:
    BrightnessLevel = namedtuple("BLevel", ['range', 'value', 'color', 'name_color'])    
    Levels = [ BrightnessLevel(range=range(0, 50), value=0, color = (255,255,255), name_color = 'Black'), 
            BrightnessLevel(range=range(49, 100), value=1, color = (243,14,255), name_color = 'Fuchsia'), 
            BrightnessLevel(range=range(99, 150), value=2, color = (154,250,0), name_color = 'Green'), 
            BrightnessLevel(range=range(149, 200), value=3, color = (243,14,255), name_color = 'Fuchsia'),
            BrightnessLevel(range=range(199, 256), value=4, color = (0,0,0), name_color = 'White')]
    

    #GETTING THE FIRST FOURTH OF THE IMAGE (UPPER LEFT), AND THE SECOND HORIZONTAL HALF:
    first_eight = img[:h//4, : w//2] 
    second_h_half = img[h//2:, :]
    # cv2.imshow("First Eight", first_eight)                             #Re-show scene image.
    # cv2.waitKey(0)                                                     #Wait for a key to be pressed in order to close the window.
    # cv2.imshow("Second Horizontal Half", second_h_half)                #Re-show scene image.
    # cv2.waitKey(0)                                                     #Wait for a key to be pressed in order to close the window.
    
    #CONVERTING THE TWO CROPPED IMAGES IN THE HSV COLOR SPACE:
    hsv_img_description = cv2.cvtColor(first_eight, cv2.COLOR_BGR2HSV)   #Converting the fourth of the image from BGR to HSV. 
    _, _, v1 = cv2.split(hsv_img_description)                            #Unpacking hue,saturation,value for the fourth of the image.
    hsv_img_plate = cv2.cvtColor(second_h_half, cv2.COLOR_BGR2HSV)       #Converting the second horizontal half of the image from BGR to HSV.
    _, _, v2 = cv2.split(hsv_img_plate)                                  #Unpacking hue,saturation,value for the second horizontal half. 
    
    #FINDING THE AVERAGE BRIGHTNESS OF THE TWO CROPPED AREA:
    avg_brightness_description = int(np.average(v1.flatten()))
    avg_brightness_plate = int(np.average(v2.flatten()))
    
    #FINDING THE BRIGHTNESS AND COLOR LEVELS:
    for level in Levels:                                                  #For each level of brightness in the possible levels of brightness (defined above).
        if avg_brightness_description in level.range:                     #If the average brightness falls in the range of the level.
            brightness_level_description = level.value                    #The Brightness Level Description is set to the value of the level. 
            color_level_description = level.color                         #The Color Level Description is set to the color of the level. 
            color_name_description = level.name_color                     #The Color Name Description is set to the level's color name of the level. 
        if avg_brightness_plate in level.range:                           #If the average brightness plate falls in the range of the level.
            brightness_level_plate = level.value                          #The Brightness Level Plate is set to the value of the level. 
            color_level_plate = level.color                               #The Color Level Plate is set to the color of the level. 
            color_name_plate = level.name_color                           #The Color Name Plate is set to the level's color name of the level.
            
    # print(f"\nThe Description Brightness Level is: {brightness_level_description} and its color associated is: {color_name_description} ")
    # print(f"\nThe Plate Brightness Level is: {brightness_level_plate} and its color associated is: {color_name_plate} ")
    return brightness_level_description, color_level_description, brightness_level_plate, color_level_plate


def Detect_and_Recognize_License_Plate():
    #This Outer function is the one that has the role to Wrap Up all the pieces of the other functions and works as a sort of Main.
    '''
    def Training_KNN_EMNIST():
        #This Inner Function is the one that has the role to load the Data we are going to use during the KNN Classification.
        
        def Loading_and_Download_Dataset_EMNIST():
            #This Inner-Inner Function is the one that has the role to load the Dataset EMNIST we are going to use during the KNN Classificationand to save the txt files.
            
            #SAVING TEXT FILES 
            def Save_txt_Files(train_labels, train_data, test_labels, test_data):
                np.savetxt('Dataset_Txt/Train_Labels.txt', train_labels, delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                np.savetxt('Dataset_Txt/Train_Data.txt', train_data, delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                np.savetxt('Dataset_Txt/Test_Labels.txt', test_labels, delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                np.savetxt('Dataset_Txt/Test_Data.txt', test_data, delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                return

            #COMPOSING SEVERAL TRANSFORMATIONS
            Transforms = transforms.Compose([ToTensor(), MyRotationTransform(270), MyFlipTransform(), transforms.Normalize((0.5,), (0.5,))])

            #SPLITTING TRAINING_DATASET AND TEST_DATASET
            training_dataset = datasets.EMNIST(root = "Dataset_Alphabet_Numbers", split = 'byclass',train = True, download = True, transform = Transforms)       
            test_dataset = datasets.EMNIST(root = "Dataset_Alphabet_Numbers",  split = 'byclass', train = False, download = True, transform = Transforms)
            
            #GETTING TRAIN_LABELS AND TRAIN_DATA 
            train_labels = training_dataset.targets
            train_data = training_dataset.data.view(697932, -1) 
            print('The Train Labels are: ', train_labels, 'with shape: ', train_labels.size())
            print('The Train Data are: ', train_data, 'with shape: ', train_data.size())
            Plotting_Dataset(training_dataset)
            
            #GETTING TEST_LABELS AND TEST_DATA 
            test_labels = test_dataset.targets
            test_data = test_dataset.data.view(116323, -1)
            print('The Test Labels are: ', test_labels, 'with shape: ', test_labels.size())
            print('The Test Data are: ', test_data, 'with shape: ', test_data.size())

            #PLOTTING_DATASET
            Plotting_Dataset(train_data, test_labels)

            #Save_txt_Files(train_labels, train_data, test_labels, test_data)
            #print(f'\nThe proportion of Test / Training data is: Test: {round((len(test_data) / len(training_data))*100)} Training: {100-(round((len(test_data) / len(training_data))*100))}' )
            return train_labels, train_data
        
        
        ########## START OF THE "Training_KNN_EMNIST" FUNCTION (Outer One) ##########

        #COMPUTE THE LABELS AND DATA DIRECTLY:
        Labels, Data = Loading_and_Download_Dataset_EMNIST()
        
        #ALTERNATIVE WAY TO LOAD THE LABELS AND DATA CLASSIFICATION FROM TXT FILES:
        #Labels = np.loadtxt("Dataset_Txt/Train_Labels.txt", np.float32)    
        #Data = np.loadtxt("Dataset_Txt/Train_Data.txt", np.float32)              #Read in training images

        #TRAINING THE KNN OBJECT:
        print('Shapes:',Data.shape, Labels.shape)
        KNN2 = KNN_sklearn(n_neighbors = 1)
        KNN2.fit(Data, Labels)
        
        #SAVING THE MODEL AS A PICKLE FILE: 
        joblib.dump(KNN2, 'Models/KNN2.pkl')
        
        return KNN2                             #If we got here, the training was successful so return true
    '''
    
    def Training_KNN_Dataset():
        #This Inner Function is the one that has the role to load the Data we are going to use during the KNN Classification.
        Labels,Data=Loading_and_Download_Dataset()          #Getting Labels and Data by calling Loadind_and_Downloading_Dataset Function. 

        #TRAINING THE KNN OBJECT:
        print('Shapes:',Data.shape, Labels.shape)
        KNN2 = KNN_sklearn(n_neighbors = 1)                 #KNN with k=1 (classifying according to the nearest neighbour).
        Data.ravel()                                        #It returns a contiguous flattened array of Data. 
        Labels.ravel()                                      #It returns a contiguous flattend array of Labels. 
        KNN2.fit(Data, Labels)                              #Fitting the KNN classifier from the training dataset Data.
        
        #SAVING THE MODEL AS A PICKLE FILE: 
        joblib.dump(KNN2, 'Models\KNN2.pkl')                #Saving the model as a pickle file.
        
        return KNN2                                         #Return the model. 

    ########## START OF THE "Detect_and_Recognize_License_Plate" FUNCTION (Outer One) ##########

    #GET THE CURRENT  DATE AND TIME:
    dt = str(datetime.datetime.now())
    print("Date and Time: ", dt[:19])

    #LOADING THE KNN-SKLEARN MODEL FROM THE FUNCTION (IT TAKES A LOT, SINCE IT DOES ALSO THE TRAINING):
    #KNN2 = Training_KNN_EMNIST()

    #ALTERNATIVE AND FASTER WAY: LOADING THE KNN-SKLEARN MODEL FROM THE FILE(ALREADY TRAINED BY US):
    KNN2 = joblib.load('Models/KNN2.pkl')
        
    #LOADING THE VIDEO/IMAGE THROUGH WHICH WE ARE GOING TO DETECT AND RECOGNIZE THE LICENSE PLATE:
    #DO THE FOLLOWING IF YOU WANT TO READ FROM A PRESENT VIDEO:
    video_input = "video1.mp4"
    #Original_Image = get_from_Video(video_input)
    
    #DO THE FOLLOWING IF YOU WANT TO READ FROM AN IMAGE:
    #Original_Image = cv2.imread('img/car1.png')  
    Original_Image = get_from_Video(video_input)
    
    #DO THE FOLLOWING IF YOU WANT TO READ FROM THE CAMERA:
    #Original_Image = get_Video(dt)
    
    #COMPUTING THE LEVEL OF BRIGHTNESS OF THE IMAGE:
    brightness_level_description, color_level_description, brightness_level_plate, color_level_plate = Get_Level_Brightness_Image(Original_Image)
     
    #COMPUTING THE LIST OF CANDIDATE LICENSE PLATES: 
    Candidate_License_Plates_L = Detect_License_Plates(Original_Image)           #It returns the List of Candidate License Plates we are going to find out in the Image 
                                                                                 #Processed.
        
    if len(Candidate_License_Plates_L) == 0:                                     #If no plates were found.
        print("\nno license plates were detected\n")                             #Inform the user that no plate was found.
        
    else: 
        
        #FIGURE OUT THE FINAL_LICENSE PLATE RECOGNITION:
        Final_License_Plate = Final_License_Plate_Extrapolation(Candidate_License_Plates_L, KNN2)        #Detect characters in plates.
        # cv2.imshow("Original_Image", Original_Image)                                                   #Show scene image.
        # cv2.waitKey(0)                                                                                 #Wait for a key to be pressed in order to close the window.
        #cv2.imshow("imgPlate", Final_License_Plate.License_Plate_Img)                                   #Show crop of plate and threshold of plate.
        #cv2.waitKey(0)                                                                                  #Wait for a key to be pressed in order to close the window.
        cv2.imshow("imgThresh", Final_License_Plate.Thresholded_Img)                                     #Showing Thresholded Image of the Final License Plate.
        cv2.waitKey(0)                                                                                   #Wait for a key to be pressed in order to close the window.

        if len(Final_License_Plate.License_Plate_String) != 0:                                           #If no chars were found in the plate.
            
            #DRAWING THE RECTANGLE AROUND THE LICENSE PLATE:
            p2fRectPoints = cv2.boxPoints(Final_License_Plate.Rotated_Rectangle_Information)             #Get 4 vertices of rotated rect.
            #Draw 4 red lines
            cv2.line(Original_Image, tuple([int(x) for x in p2fRectPoints[0]]), tuple([int(x) for x in p2fRectPoints[1]]), color_level_plate, 2)         
            cv2.line(Original_Image, tuple([int(x) for x in p2fRectPoints[1]]), tuple([int(x) for x in p2fRectPoints[2]]), color_level_plate, 2)
            cv2.line(Original_Image, tuple([int(x) for x in p2fRectPoints[2]]), tuple([int(x) for x in p2fRectPoints[3]]), color_level_plate, 2)
            cv2.line(Original_Image, tuple([int(x) for x in p2fRectPoints[3]]), tuple([int(x) for x in p2fRectPoints[0]]), color_level_plate, 2)
            
            ############ HERE WE WILL GET THE DATA FROM THE DATABASE: (We have not implemented the DB solution here since we will not use it. The DB is implemented below 
            #                                                          when we use the CNN!!!)
            ID_Employee = '_'                                                                             #Getting the ID of the Employee
            Name_Employee = ''                                                                            #Getting the Name of the Employee
            Surname_Employee = ''                                                                         #Getting the Surname of the Employee
            ######################################
            cv2.putText(Original_Image, 'Date & Time: '  + dt[:19], (5, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_level_description, 1, cv2.LINE_8)      #Inserting text 'Date and Time' 
            cv2.putText(Original_Image, 'License Plate Recognized: ' + Final_License_Plate.License_Plate_String, (5, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_level_description, 1, cv2.LINE_8)     #Inserting text 'License Plate Recognized'
            cv2.putText(Original_Image, f'ID Employee Recognized: {ID_Employee}', (5, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_level_description, 1, cv2.LINE_8)        #Inserting text 'ID Employee Recognized'
            cv2.putText(Original_Image, 'Company X', (5, 75), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_level_description, 1, cv2.LINE_8)                     #Inserting text 'Company X'
            
            #PRINT OUT ON COMMAND LINE THE LICENSE PLATE RECOGNIZED:
            print("\nLicense Plate Recognized: " + (Final_License_Plate.License_Plate_String).upper() + "\n----------------------------------------")  # write license plate text to std out

            #WRITING THE LICENSE PLATE DIRECTLY ON THE IMAGE:
            #GETTING THE HEIGHT OF THE ENTIRE IMAGE AND OF THE LICENSE PLATE, RESPECTIVELY:
            h_img, _, _ = Original_Image.shape                                          #Getting height,width,number of channels of the Entire image 
            h_license_plate, _, _ = Final_License_Plate.License_Plate_Img.shape         #Getting height,width,number of channels of the final license plate image 

            #GETTING THE TEXT_SIZE OF THE LICENSE PLATE, GETTING THE BOUNDING BOX OF THE TEXT STRING.
            (w_text, h_text), _ = cv2.getTextSize((Final_License_Plate.License_Plate_String).upper(), cv2.FONT_HERSHEY_TRIPLEX , float(h_license_plate) / 30.0 , int(round((float(h_license_plate) / 30.0) * 1.5)))        # call getTextSize

            #GETTING THE COM(CENTER OF MASS) OF THE LICENSE PLATE:
            (x_center_lplate, y_center_lplate), _, _  = Final_License_Plate.Rotated_Rectangle_Information
            
            #RELOCATING THE PLATE TEXT IN THE FRAME IMAGE:
            #CHECKING WHERE THE PLATE IS W.R.T. THE IMAGE's HEIGHT: 
            if y_center_lplate < (h_img * 0.75):                                                  #If the license plate is in the upper 3/4 of the image
                y_center_lplate = y_center_lplate + (h_license_plate * 1.6)                       #Write characters below the plate
            else:                                                                                 #Else,if the license plate is in the lower 1/4 of the image
                y_center_lplate = y_center_lplate - (h_license_plate * 1.6)                       #Write characters above the plate

            #COMPUTING THE LEFT BOTTOM POINT:
            LB_point = (  int(  x_center_lplate - (w_text / 2)  ),  int(  y_center_lplate + (h_text / 2)  ) )

            #WRITING ON THE LICENSE PLATE THE LICENSE PLATE:
            cv2.putText(Original_Image, (Final_License_Plate.License_Plate_String).upper(), LB_point, cv2.FONT_HERSHEY_TRIPLEX, float(h_license_plate) / 30.0, color_level_plate, int(round((float(h_license_plate) / 30.0) * 1.5)))
                    
                    
            #SHOWING THE FINAL IMAGE WITH THE RECOGNIZED LICENSE PLATE:
            cv2.imshow("Automatic License Plate Recognition (ALPR)", Original_Image)                #Re-show scene image.
            cv2.waitKey(0)                                                                          #Wait for a key to be pressed in order to close the window.
            
            #SAVING THE IMAGE
            cv2.imwrite(f"Saved_img/License_Plate_Recognition {dt[:19]}.png", Original_Image)       #Write image out to file.
        else:
            print("\nNo License Plate Detected \n\n")                                               #If the License Plate has lenght zero, No license plate has been detected. 
            return
    return

class Custom_Dataset(Dataset):
    '''Dataset Class for extrapolating by hand the tuple image-label'''
    def __init__(self, data, labels, l):
        super().__init__()
        self.labels = labels                               #Initializing the labels. 
        self.data = torch.tensor(data).view(l,1,28,28)     #Initializing the data as a tensor of shape  l (batch size),1,28,28.
    def __len__(self):
            return len(self.labels)                        #It returns the lenght of the labels. 
    def __getitem__(self, idx):
            label = self.labels[idx]
            data = self.data[idx]
            sample = (data, label)                         #A sample is a tuple composed of data,label. 
            return sample

class Licence_Plate_CNN(nn.Module):
    '''Class for representing the License Plate CNN'''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)             #Starting Layer of the CNN. Note that we have to give the input channel, output channels and the Kernel Size.
        self.conv2 = nn.Conv2d(6, 10, 3)            #The second Layer of our CNN will have to take in input channels the output channel of the prevoius layer.
        
        self.fc1 = nn.Linear(24*24*10, 20)          #Defining the Fully-Connected layer.
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 36)                #Final Layer, outputting 36 corresponding to number of classes-1.
        self.relu = nn.Mish()                       #We have used Mish as our activation function. Mish has a stronger theoretical pedigree, and in testing delivers, 
                                                    #on average, superior performance over ReLU in terms of both training stability and accuracy.

    #Defining the forward function. Note that here, we will have to specify how to connection among layers, since we have not used the Sequential approach. 
    def forward(self, x):
        #It says how x passes through all these layers
        #First Convolution.
        x = self.conv1(x)
        x = self.relu(x)                            #Applying the Activation Function.
            
        #Second Convolution.
        x = self.conv2(x)
        x = self.relu(x)                            #Applying the Activation Function. 
            
        #Fully Connected
        x = torch.flatten(x, 1)                     #Flatten all dimensions except the batch. We have to put "1" beacuse the input data is a tensor of 4 elements
                                                    #[batch,chanels, width, height]. Flatten the data starting from index one.
        #FC1
        x = self.fc1(x)
        x = self.relu(x)                            #Applying the Activation Function.
            
        #FC2
        x = self.fc2(x)
        x = self.relu(x)                            #Applying the Activation Function.
            
        #FC out
        x = self.fc3(x)
        #We don't need to use the mish. It will provide an array that has the same length of the classes.
            
        return x
    
    
device='cuda' if torch.cuda.is_available() else 'cpu'                       #Defining the device for the computation.
torch.manual_seed(42)                                                       #Setting a seed, in order to allow reproducibility. 

def Detect_and_Recognize_License_Plate_CNN():
    Labels,Data=Loading_and_Download_Dataset()                              #Getting Labels and Data by calling Loadind_and_Downloading_Dataset Function.
    df= pd.DataFrame({'Data': Data.tolist(), 'Labels': Labels.tolist()})    #Define the Dataframe. 
    ds= Custom_Dataset(df['Data'], df['Labels'], len(df['Labels']))         #Define the dataset object. 
    
    def sampleFromClass(ds, k):
        '''It will return the two Train and Test Dataset'''
        #This function performs a train-test manual split
        class_counts = {}                                                   #Initializing a dictionary. 
        train_data = []                                                     #Initializing the train data list. 
        train_label = []                                                    #Initializing the train label list. 
        test_data = []                                                      #Initializing the test data list. 
        test_label = []                                                     #Initializing the tets label list. 
        for i in range(len(ds)):
            label=torch.tensor([ds.labels[i]])
            data=torch.tensor(ds.data[i])
            c = ds.labels[i]
            class_counts[c] = class_counts.get(c, 0) + 1
            if class_counts[c] <= k:
                train_data.append(data)                                     #Append data to the train_data list. 
                train_label.append(label)                                   #Append label to the train_label list. 
            else:                                                           #While, if it is less than k.  
                test_data.append(data)                                      #Append data to the test_data. 
                test_label.append(label)                                    #Append label to the test_label. 
        train_data = torch.cat(train_data)                                  #It concatenates the train_data, defined as tensors above. 
        train_label = torch.cat(train_label)                                #It concatenates the train_label, defined as tensors above.
        test_data = torch.cat(test_data)                                    #It concatenates the test_data, defined as tensors above.
        test_label = torch.cat(test_label)                                  #It concatenates the test_label, defined as tensors above. 

        return (Custom_Dataset(train_data, train_label, len(train_label)), 
            Custom_Dataset(test_data, test_label, len(test_label)))
        
  
    def train_and_test(ds):
        #The train_and_test function is made of two inner functions, defining respectively the Training and Testing Loop.
        def trainingLoop(train_dataloader, model, loss_fn, optimizer):
            '''Training Loop Function'''
            print_size = len(train_dataloader.dataset)
        
            for batch, (X,y) in enumerate(train_dataloader):
                pred= model(X)                                              #Getting the prediction.
                loss = loss_fn(pred,y)                                      #Distance with the batch containing the input image.

                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch%100==0:                                            #Every 100 batches, we store our reconstruction.
                    loss= loss.item()
                    print(f'The Current Loss is: {loss}')
        
        def testLoop(test_dataloader, model, loss_fn):
            '''Test Loop Function'''
            print_size = len(test_dataloader.dataset)                                 
            num_batches = len(test_dataloader)                                         #It is the lenght of the test_dataloader. 
            test_loss = 0                                                              #Intialize the test loss to zero. 
            correct = 0                                                                #Iniziatile the number of correct labels to zero.
            
            with torch.no_grad():                                                      #Do not modify the weights of the model.
                for X,y in test_dataloader:
                    X, y = X.to(device), y.to(device)
                    
                    pred = model(X)                                                    #Getting the prediction.
                    test_loss += loss_fn(pred, y).item()                               #With the ".item", we just get the value of the loss.
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    
                test_loss = test_loss / num_batches                                    #The test loss is defined as the value of the test loss divided by the number of 
                                                                                       #batches. 
                correct = correct / print_size                                         #Correct is defined as the value of correct divided by print_size.
                print(f"The Current Accuracy is: {correct * 100}, with the Average Loss being:{test_loss}") 
        
        train_dataset,test_dataset=sampleFromClass(ds,711)                             #Extracting Train_dataset and test_dataset by calling the 'manual split' function.
        train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True)   #DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
        test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = True)     #The batch size takes 8 samples at each iteration,until the entire dataset has been seen.
        
        #instance of our model
        model=Licence_Plate_CNN().to(device)

        #hyperparameter settings
        learning_rate= 1e-3
        epochs= 10

        #loss function definition
        loss_fn =nn.CrossEntropyLoss()

        #optimizer definition
        optimizer = torch.optim.Adam(model.parameters(),learning_rate)  
        
                
        #model = torch.load('Models/CNN_7_Version_92.194.pt')                           #At this point, we load the model. This acts as a sort of weight, allowing us to start 
                                                                                        #doing the training and testing from a better accuracy. Thanks to it, we were able to 
                                                                                        #reach an accuracy near to 96% .. It was a big improvement, but not all the license 
                                                                                        #plates were recognized correctly. Reaching a higher accuracy may cause overfitting 
                                                                                        #problems. 
        for e in range(epochs):                                                         #For each epoch. 
            trainingLoop(train_dataloader,model,loss_fn,optimizer)                      #Call the Training Loop. 
            testLoop(test_dataloader, model, loss_fn)                                   #Call the Test Loop. 
        torch.save(model,'Models/CNN.pt')                                               #Save the model. 
        return model
    

    ########## START OF THE "Detect_and_Recognize_License_Plate_CNN" FUNCTION (Outer One) ##########

    #GET THE CURRENT  DATE AND TIME:
    dt = str(datetime.datetime.now())
    timestamp = dt[:19]
    print("Date and Time: ", timestamp)
    #CNN=train_and_test(ds)

    #LOAD THE CNN WE ARE USING: 
    #CNN=torch.load('Models/CNN_5_Version_91.439.pt')
    CNN=torch.load('Models/CNN_6_Version_91.539.pt')
    #CNN=torch.load('Models/CNN_7_Version_92.194.pt')
    #CNN=torch.load('Models/CNN_8_Version_95.628.pt')

    #LOADING THE VIDEO/IMAGE THROUGH WHICH WE ARE GOING TO DETECT AND RECOGNIZE THE LICENSE PLATE:
    #DO THE FOLLOWING IF YOU WANT TO READ FROM A PRESENT VIDEO:
    #video_input = "Video/Get_Video_Recognition2.mp4"
     
    #DO THE FOLLOWING IF YOU WANT TO READ FROM AN IMAGE:
    Original_Image = cv2.imread('img/car4.png') 
    #Original_Image = get_from_Video(video_input) 
    
    #COMPUTING THE LEVEL OF BRIGHTNESS OF THE IMAGE:
    brightness_level_description, color_level_description, brightness_level_plate, color_level_plate = Get_Level_Brightness_Image(Original_Image)
     
    #COMPUTING THE LIST OF CANDIDATE LICENSE PLATES: 
    Candidate_License_Plates_L = Detect_License_Plates(Original_Image)        #It returns the List of Candidate License Plates we are going to find out in the Image Processed.
    
    if len(Candidate_License_Plates_L) == 0:                                  #If no plates were found.
        print("\nno license plates were detected\n")                          #Inform the user that no plate has been found. 
        
    else: 
        
        #FIGURE OUT THE FINAL_LICENSE PLATE RECOGNITION:
        Final_License_Plate = Final_License_Plate_Extrapolation(Candidate_License_Plates_L, CNN)        #Detect character in plates.
        # cv2.imshow("Original_Image", Original_Image)                                                  #Show scene image.
        # cv2.waitKey(0)                                                                                #Wait for a key to be pressed in order to close the window.
        #cv2.imshow("imgPlate", Final_License_Plate.License_Plate_Img)                                  #Show crop of plate and threshold of plate.
        #cv2.waitKey(0)                                                                                 #Wait for a key to be pressed in order to close the window.
        cv2.imshow("imgThresh", Final_License_Plate.Thresholded_Img)                                    #Show thresholded image. 
        cv2.waitKey(0)                                                                                  #Wait for a key to be pressed in order to close the window.

        Recognition = True
        if len(Final_License_Plate.License_Plate_String) != 0:                  #If the length of the license plate string is different from zero. 
            
            #DRAWING THE RECTANGLE AROUND THE LICENSE PLATE:
            p2fRectPoints = cv2.boxPoints(Final_License_Plate.Rotated_Rectangle_Information)            #Get 4 vertices of rotated rectangle 
            cv2.line(Original_Image, tuple([int(x) for x in p2fRectPoints[0]]), tuple([int(x) for x in p2fRectPoints[1]]), color_level_plate, 2)         #Draw 4 red lines
            cv2.line(Original_Image, tuple([int(x) for x in p2fRectPoints[1]]), tuple([int(x) for x in p2fRectPoints[2]]), color_level_plate, 2)
            cv2.line(Original_Image, tuple([int(x) for x in p2fRectPoints[2]]), tuple([int(x) for x in p2fRectPoints[3]]), color_level_plate, 2)
            cv2.line(Original_Image, tuple([int(x) for x in p2fRectPoints[3]]), tuple([int(x) for x in p2fRectPoints[0]]), color_level_plate, 2)
            
            ############ HERE WE WILL GET THE DATA FROM THE DATABASE:
            try:
                with psycopg2.connect( host = hostname, dbname = database, user = username, password = pwd, port = port_id) as conn:    #Establishing a connection with postgres

                    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:    #creation of a cursor during the connect to execute the query. 

                        query = 'SELECT c."ID_Employee", e."Name_Employee", e."Surname_Employee" \
                                             FROM "CAR" as c JOIN "EMPLOYEE" as e ON c."ID_Employee" = e."ID_Employee"\
                                             WHERE c."License_Plate" = ' + "'" + '%s' % (Final_License_Plate.License_Plate_String).upper() + "'"
                        print('\nLicense Plate Recognition...\n')
                        
                        try:
                            cur.execute(query)         #It tries to execute the query. 
                            result = cur.fetchall()    #It fetches all the rows of a query result. It returns all the rows as a list of tuples.
                            df = pd.DataFrame(result)  #Dataframe 
                            df.columns = ['ID_Employee', 'Name_Employee', 'Surname']  #Columns of the Dataframe. 
                            print("ALPR Succeeded!")
                            print(df)
                            ID_Employee = f"{df['ID_Employee'].iat[0]}"
                            Name_Employee = f"{df['Name_Employee'].iat[0]}"
                            Surname_Employee = f"{df['Surname'].iat[0]}"
                                
                        except (Exception, psycopg2.DatabaseError) as ex:     #The program will enter in the except if the license plate has not been detected correctly or if 
                                                                              #it is not stored in the database (..someone who is not part of the Company is trying to enter).
                            print('Employee License Plate not Recognized.')
                            ID_Employee = f"NOT RECOGNIZED"   
                            Recognition = False             #It will be used later.. 
                        
                        insertion_query = 'INSERT INTO "PRESENCE" ("ID_Presence", "Typology_Presence", "Date_Time_Presence", "ID_Employee") VALUES (%s,%s,%s,%s)'   #Insert Presence  
                        presence = "AMG_Presence_" + timestamp[:10] + '_' + timestamp[11:]   
                        record_to_insert = (presence, 'Entry', timestamp, ID_Employee)          #Record to be inserted into Presence
                        cur.execute(insertion_query, record_to_insert)
    
            except Exception as error:     #The program gets here if a connection has not been established  
                print(error)
            finally:                        
                if conn is not None:       #If the connection has been established 
                    conn.close()           #Close the connection!
        
            
            ######################################
            #PRINT OUT ON COMMAND LINE THE LICENSE PLATE RECOGNIZED:
            print("\nLicense Plate Recognized: " + (Final_License_Plate.License_Plate_String).upper() + "\n----------------------------------------")  

            #WRITING THE LICENSE PLATE DIRECTLY ON THE IMAGE:
            #GETTING THE HEIGHT OF THE ENTIRE IMAGE AND OF THE LICENSE PLATE, RESPECTIVELY:
            h_img, _, _ = Original_Image.shape                                      #Getting the height, width, number of channels of the Entire image 
            h_license_plate, _, _ = Final_License_Plate.License_Plate_Img.shape     #Getting the height, width, number of channels of the License Plate 

            #GETTING THE TEXT_SIZE OF THE LICENSE PLATE, GETTING THE BOUNDING BOX OF THE TEXT STRING.
            (w_text, h_text), _ = cv2.getTextSize((Final_License_Plate.License_Plate_String).upper(), cv2.FONT_HERSHEY_TRIPLEX , float(h_license_plate) / 30.0 , int(round((float(h_license_plate) / 30.0) * 1.5)))        # call getTextSize

            #GETTING THE COM(CENTER OF MASS) OF THE LICENSE PLATE:
            (x_center_lplate, y_center_lplate), _, _  = Final_License_Plate.Rotated_Rectangle_Information
            
            #RELOCATING THE PLATE TEXT IN THE FRAME IMAGE:
            #CHECKING WHERE THE PLATE IS W.R.T. THE IMAGE's HEIGHT: 
            if y_center_lplate < (h_img * 0.75):                                                  #If the license plate is in the upper 3/4 of the image
                y_center_lplate = y_center_lplate + (h_license_plate * 1.6)                       #Write characters below the plate
            else:                                                                                 #Else,if the license plate is in the lower 1/4 of the image
                y_center_lplate = y_center_lplate - (h_license_plate * 1.6)                       #Write characters above the plate

            #COMPUTING THE LEFT BOTTOM POINT:
            LB_point = (  int(  x_center_lplate - (w_text / 2) ),  int(  y_center_lplate + (h_text / 2)  ) )

            #WRITING ON THE LICENSE PLATE THE LICENSE PLATE:
            ##### Two unfortunate situations may occur : The CNN will not recognize the license plate well or the person trying to enter is not an employee of the Company 
            if Recognition == False:     #If No Recognition
                Final_License_Plate.License_Plate_String = "NOT RECOGNIZED!"    #"NOT RECOGNIZED" will be displayed 
                color_level_plate = (0,0,255)           #The color level plate is set to red (as a warning) 
                color_level_description = (0,0,255)     #The color level description is set to red(as a warning)
                LB_point = (LB_point[0] -140, LB_point[1])    #Performing some adjustements  
            
            #In the upper left part, the following strings will be displayed 
            cv2.putText(Original_Image, (Final_License_Plate.License_Plate_String).upper(), LB_point, cv2.FONT_HERSHEY_TRIPLEX, float(h_license_plate) / 30.0, color_level_plate, int(round((float(h_license_plate) / 30.0) * 1.5)))
            cv2.putText(Original_Image, 'Date & Time: '  + timestamp, (5, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_level_description, 1, cv2.LINE_8)
            cv2.putText(Original_Image, 'License Plate Recognized: ' + Final_License_Plate.License_Plate_String, (5, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_level_description, 1, cv2.LINE_8)
            cv2.putText(Original_Image, f'ID Employee Recognized: {ID_Employee}', (5, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_level_description, 1, cv2.LINE_8)
            cv2.putText(Original_Image, 'Company AMG', (5, 75), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_level_description, 1, cv2.LINE_8) 
                    
            #SHOWING THE FINAL IMAGE WITH THE RECOGNIZED LICENSE PLATE:
            cv2.imshow("Automatic License Plate Recognition (ALPR)", Original_Image)                #Re-show scene image
            cv2.waitKey(0)                                                                          #Wait for a key to be pressed in order to close the window.
            
            #SAVING THE IMAGE
            cv2.imwrite(f"Saved_img/License_Plate_Recognition {dt[:19]}.png", Original_Image)           #Write image out to file
        else:
            print("\nNo License Plate Detected \n\n")           #If the lenght of the license plate is zero,no license plate has been detected 
            return
    return
if __name__ == "__main__":
    Detect_and_Recognize_License_Plate_CNN()