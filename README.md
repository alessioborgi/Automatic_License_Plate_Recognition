# Automatic_License_Plate_Recognition (ALPR)

Copyright © 2022 Alessio Borgi

PROJECT SCOPE: Automatic License Plate Recognition of a given Image/Video through ML and DL Techniques dedicated to Companies that would like to keep track and automatize gate accesses.

PROJECT RESULTS: 
• Possibility to process three different typologies of recognition inputs: Images, Pre-Registered Videos, and Live-Videos.
• Recognition of License Plates. After Recognition, PostgreSQL Database Query in order to check License Plate Validity.
• Additional Implementation of a Customer GUI(that is entering the gate) that explains to him/her all the License Plate Recognition processes and provides Feedback on the recognition status.
• Additional Implementation of a Company Manager GUI through which the Company Security Manager can check for accesses to the Company Building.  
• In the meanwhile, possibility to have Employee's work hours registration. 

PROJECT DETAILS: 
• Image Pre-Processing in order to guarantee a better input for the CNN. (Passage through HSV Color Space, Application of Black-Hat and Top-Hat Morphological Transformations, De-Noising Techniques and Adaptive Thresholding).
• License Plate Detection as a second step, given the pre-processed input. 
• License Plate Recognition given the Detected License Plate. (Division of the License Plate in ROIs, each of which containing a single character and then Recognition through a Convolutional Neural Network.)
• CNN Implementation using Cross-Entropy Loss, MISH Activation Function, Training over 100 epochs. Model Peak Reached: 95% Accuracy. Chosen Model Accuracy: 91% (In order to avoid Overfitting).
• For the License Plate Recognition, tentative application of the KNN ML Algorithm (failed). 
• Customized Dataset Implementation based on the folder hierarchy. Manual Split in order to have a Balanced Train-Test.

Project Repository: https://github.com/alessioborgi/Automatic_License_Plate_Recognition
