'''
PARENT:
    copy2_spline_curve_generator_path_finder_motion_race_distractor_many_no_occlusion_fixed_blob_256_permanent_markers.py
CHANGELOG:
    - Same resolution - at 256
        - Same $\eta_{unique}$ ranges
        - Same intervals of separation
    - Changed rings and circles to squares
'''

import time 

import numpy as np
# import math
import random
import argparse
import os
import skimage  # try and use only the draw functions, and not the entire library
import imageio  # can be eliminated altogether if not writing images, just dumping np array
# from scipy.sparse import coo_matrix # make sparse matrix for individual frames, and then write to a bigger np.ndarray 

from spline_scaled_curve_generator_class import GenCoord

#Function to work with multiple boolean inputs
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Basic commandline argument parsing 
parser = argparse.ArgumentParser()
   
parser.add_argument('-r', '--radius', type=int, default=1, help="int: height and width of the squares in stimuli [default 1]")
parser.add_argument('-n', '--num_samples', type=int, default=10, help="int: number of samples to be generated [default 10]")
parser.add_argument('-ss', '--start_sample', type=int, default=0, help="int: number at which to start the sample numbering [default 0]")
parser.add_argument('-nd', '--num_distractors', type=int, default=10, help="int: number of distractor paths [default 10]")
parser.add_argument('-pl', '--path_length', type=int, default=64, help="int: length of the paths [default 64]")
parser.add_argument('-ed', '--extra_dist', type=int, default=4, help="int: number of extra distractors to be added to the movie [defalut 4]")
parser.add_argument('-sp', '--skip_param', type=int, default=1, help="int: (slice step) number of coordinates to skip from the generated set of full coordinates. Increases the speed of points, by increasing their path length (displacement) but keeping the number of frames same (time). MIN: 1. MAX: 5 [defalut 1]")
parser.add_argument('-HM', '--HUMAN_MODE', type=str2bool, nargs='?', const=True, default=False, help="bool: Activate human mode. Show path lines for all paths and slow down the movie with extra frames [default False]")
parser.add_argument('-NS', '--NEGATIVE_SAMPLE', type=str2bool, nargs='?', const=True, default=False, help="bool: Generate negative sample [default False]")
parser.add_argument('-g', '--gif', type=str2bool, const=True, nargs='?', default=False, help="bool: Generate movie frames in gif files, and atore it with the frames [default False]")
parser.add_argument('-si', '--save_image', type=str2bool, const=True, nargs='?', default=False, help="bool: Store individual frames on disk in individual directories at the specified path. [default False]")
parser.add_argument('-p', '--path', type=str, default=os.getcwd(), help="str: Path to store the folder structure of the generate samples [default current directory]")

args = parser.parse_args()

skip_param=args.skip_param


##########################################################################################
##                  Drawing functions
##########################################################################################

def rectangle(r0, c0, width, height):
    #draws a filled in rectangle
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon(rr, cc)


def rectangle_perimeter(r0, c0, width, height, shape=None, clip=False):
    #draws only the perimeter of the rectangle
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)



##########################################################################################
##                  Coordinate generation logic
##########################################################################################







def get_points(nPoints):
    #if seeded, it will return the exact same points for all curves
    #np.random.seed(999)
    
    return np.random.rand(nPoints,2)*200

def check_range_set(s, number, interval=2):
    '''
    Checks if the number is in the set
    along with the intervals on either
    side of the number line.
    For eg: if number is 10, and interval
    is 2, then this will check if the
    numbers 8,9,10,11,12 are in the given set
    or not.

    Returns: True if the range is in set
    False otherwise
    '''
    #print('in check_range_set')
    ret=False
    if interval:
        # for non zero intervals
        for i in range(number-interval,number+interval+1):
            if i in s:
                ret=True
        return ret
    else:
        # for zero interval
        if number in s:
            ret=True
        return ret
    #return ret
    

def get_full_length_coordinates(nPoints, nTimes, normalized, path_length, skip_param=skip_param):
    '''
    Returns the floor of coordinates of full length for
    both the tracking curve, and the distractor
    curve of the same length.
    The coordinates are normalized to a different
    location of the screen, randomly.
    0 - no normalization (actually in the first quadrant)
    2 - normalized to the center of the screen (actually in the third quadrant)
    4 - normalized to the first quadrant of the screen (actually in the center of the display)
    6 - roughly in the center of the display
    8 - roughly in the center of the display

    The below two coordinate generation codes can be put in a loop.
    '''
    
    #normal = np.random.choice([0,2,4,6,8],2,replace=False)
    # generating unique normalization positions, which would be used to shift
    # the dots by certain number of pixels
    '''
    # DO NOT GENERATE NORMALIZED COORDINATES. NO NEED HERE. LET TARGETS BE WITHIN THE SCREEN RANGE 
    normal=[]
    for i in range(2):
        _normal = np.random.choice(range(5))#,2,replace=False) # *** changed from 5
        while check_range_set(normalized,_normal,interval=2):# _normal in normalized:
            _normal = np.random.choice(range(13))#,2,replace=False) # *** 13
        normal.append(_normal)
    #print('done with range mask\ngenerating coordinates')
    '''
    
    #generating coordinates for the positive sample with blob at both ends
    #xvals, yvals = bezier_curve(get_points(nPoints), nTimes)
    #coordinates_2 = [(xvals[i],yvals[i]) for i in range(0, nTimes)]
    
    #generating coordinates for the positive sample with blob at both ends
    #using the coordinates generator class

    '''
    #Generate 3x the coordinates, and then sample path_length out of them
    '''
    cd_2=GenCoord()
    coordinates_2=cd_2.get_coordinates(length_curve=nTimes, angle_range=[90,100], distance_points=.002, delta_angle_max=90,wiggle_room=.95,rigidity=.95, x_min=0,x_max=win_x,y_min=0,y_max=win_y)

    #taking one-third points from a random location
    rn_l=np.random.randint(len(coordinates_2)-(path_length*skip_param)+1)
    coordinates_2=coordinates_2[rn_l:rn_l+(path_length*skip_param):skip_param]
##    del coordinates_2[1]
##    del coordinates_2[1]
##    del coordinates_2[-2]
##    del coordinates_2[-2]
##    del coordinates_2[-2]
##    del coordinates_2[-2]
    #normalizing to an arbitrary location in the screen
    # NEW: shifting the dots by a random number of pixels
    normal=[0,0]
    if normal[0]:
        #coordinates_2=[((coordinates_2[i][0]-(win_x/float(normal[0]))),(coordinates_2[i][1]-(win_y/float(normal[0])))) for i in range(0,len(coordinates_2))]
        coordinates_2=[((coordinates_2[i][0]-(float(normal[0]))),(coordinates_2[i][1]-(float(normal[0])))) for i in range(0,len(coordinates_2))]



    #generating coordinates for the negative distractor sample of same length
    #xvals, yvals = bezier_curve(get_points(nPoints), nTimes)
    #coordinates_3 = [(xvals[i],yvals[i]) for i in range(0, nTimes)]

    #generating coordinates for the negative distractor sample of same length
    #using the coordinates generator class
    '''
    #Generate 3x the coordinates, and then sample path_length out of them
    '''
    cd_3=GenCoord()
    coordinates_3=cd_3.get_coordinates(length_curve=nTimes, angle_range=[90,100], distance_points=.002, delta_angle_max=90,wiggle_room=.95,rigidity=.95, x_min=0,x_max=win_x,y_min=0,y_max=win_y)

    #taking one-third points from a random location
    rn_l=np.random.randint(len(coordinates_3)-(path_length*skip_param)+1)
    coordinates_3=coordinates_3[rn_l:rn_l+(path_length*skip_param):skip_param]
##    del coordinates_3[1]
##    del coordinates_3[1]
##    del coordinates_3[-2]
##    del coordinates_3[-2]
##    del coordinates_3[-2]
##    del coordinates_3[-2]

    
    #normalizing to an arbitrary location in the screen
    if normal[1]:
        #coordinates_3=[((coordinates_3[i][0]-(win_x/float(normal[1]))),(coordinates_3[i][1]-(win_y/float(normal[1])))) for i in range(0,len(coordinates_3))]
        coordinates_3=[((coordinates_3[i][0]-(float(normal[1]))),(coordinates_3[i][1]-(float(normal[1])))) for i in range(0,len(coordinates_3))]


    #add the normalization coefficients to the normalized set for further use
    normalized.add(normal[0])
    normalized.add(normal[1])

    return np.floor(coordinates_2), np.floor(coordinates_3), normalized


def get_third_length_distractor_coordinates(nPoints, nTimes, normalized, num_distractors, path_length, skip_param=skip_param):
    '''
    Returns the coordinates of one-third length for
    specified number of distractor curves.
    The one-third length of the distractors is
    randomly sampled, from a random starting point to
    the next available 50 points.
    
    The coordinates are normalized to a different
    location of the screen, randomly, other than what
    is in normalized set already
    0 - no normalization (actually in the first quadrant)
    2 - normalized to the center of the screen (actually in the third quadrant)
    4 - normalized to the first quadrant of the screen (actually in the center of the display)
    '''

    dis_len=int(nTimes) #length of the distractor. By default one-third for this function
    coord_d=[]
    #nm_list=list(normalized)
    #nm_r=set(range(0,17))-normalized
    #normal=np.random.choice(list(nm_r), num_distractors)
    
    # generating unique normalization positions, which would be used to shift
    # the dots by certain number of pixels
    
    '''
    # DO NOT GENERATE RANDOM NUMBERS. USE THE ONES PASSED THROUGH NORMALIZED
    normal=[]
    for i in range(num_distractors):
        _normal = np.random.choice(range(-13,13))#,2,replace=False) # *** -13 to 13
        #while _normal in normalized:
        while check_range_set(normalized,_normal,interval=1):# _normal in normalized:
            _normal = np.random.choice(range(-13,13))#,2,replace=False) # *** -13 to 13
        #print(i)
        normal.append(_normal)
    '''
    normal=normalized

    for i in range(num_distractors):
        #generating coordinates for the negative distractor sample of same length for now
        #xvals, yvals = bezier_curve(get_points(nPoints), nTimes)
        #coordinates_d = [(xvals[i],yvals[j]) for j in range(0, nTimes)]

        #generating coordinates for the negative distractor sample of same length for now
        #using the coordinates generator class
        '''
        #Generate 3x the coordinates, and then sample one-third out of them
        '''
        cd_d=GenCoord()
        coordinates_d=cd_d.get_coordinates(length_curve=nTimes, angle_range=[90,100], distance_points=.002, delta_angle_max=90,wiggle_room=.95,rigidity=.95, x_min=0,x_max=win_x,y_min=0,y_max=win_y)
        
        #taking one-third points from a random location
        rn_l=np.random.randint(len(coordinates_d)-(path_length*skip_param)+1)
        coordinates_d=coordinates_d[rn_l:rn_l+(path_length*skip_param):skip_param]
        #normalizing to an arbitrary location in the screen
        # NEW: shifting the dots by a random number of pixels
        if normal[i]:
            #coordinates_d=[((coordinates_d[j][0]-(win_x/float(normal[i]))),(coordinates_d[j][1]-(win_y/float(normal[i])))) for j in range(0,len(coordinates_d))]
            coordinates_d=[((coordinates_d[j][0]-(float(normal[i]))),(coordinates_d[j][1]-(float(normal[i])))) for j in range(0,len(coordinates_d))]
        coord_d.append(coordinates_d)
        # normalized.add(normal[i])
        normalized=np.delete(normalized,0)



    return coord_d, normalized

def draw_blobs(coordinates, height, width):
    '''
    Generic function to draw blobs, given the coordinates
    No need to pass the whole set of coordinates_2 and coordinates_3, 
    just the starting and ending points depending on positive or negative sample.

    Condition to check positive/negative sample takes care of what needs to be parsed in this function.
    '''

    # rr, cc = rectangle_perimeter(np.floor(coordinates_2[0][0]), np.floor(coordinates_2[0][1]), 3, 3)
    rr, cc = rectangle_perimeter(coordinates[0], coordinates[1], height, width)

    return rr,cc 


    
    
nPoints = 2 #no. of points to be fit using the bernstein polynomial (degree of the polynomial to fit)
nTimes=22#10#16#150 #no. of points in the curve 


#if using circle from visual stimulus, use radius
# radius=1#1.5#3#5
radius=args.radius
#dimensions of window
win_x=32-5#460
win_y=32-5#460

square_height = 2-1
square_width = 2-1
blob_height = 3-1
blob_width = 3-1
channels = 3

# parameter to extend the window size beyond
# what is used for processing
# win_add = 14#250

#number of distractors
# num_distractors=10
num_distractors=args.num_distractors

# HUMAN_MODE=False # to be enabled when generating displays for human demonstration (enables things like 3..2..1 between frames and other things - to be added)
# NEGATIVE_SAMPLE=False # to be enabled when generating negative examples 
HUMAN_MODE=args.HUMAN_MODE
NEGATIVE_SAMPLE=args.NEGATIVE_SAMPLE
path_length = args.path_length #same as number of frames. To be taken as argparse argument


start_sample=args.start_sample
num_samples=args.num_samples

# images_all=np.ndarray([num_samples-start_sample,path_length,128,128,channels], dtype=np.uint8)


# if args.save_image:
# check if the directory for sample exists, else make one for positive (1) or negative (0)
# make the directory in any case, given the bigger npz/tfrecords files would also be stored in their respective sample directories
path=args.path+"/"+str(int(not args.NEGATIVE_SAMPLE))
if not os.path.exists(path):
    os.makedirs(path)

for sample in range(start_sample,num_samples):

    # start=time.time()
    

    # make individual directory for the given sample
    if args.save_image:
        path_to_save=path+"/"+str(int(not args.NEGATIVE_SAMPLE))+"_sample_"+str(sample)+"/"
        # print("Saving to: "+path_to_save)
        try:
            os.makedirs(path_to_save)
        except OSError:
            pass

    points = get_points(nPoints)
    normalized=set([])





    #coordinates = [(xvals[i],yvals[i]) for i in range(nTimes)]
    #normalizing to the center of the screen
    #coordinates=[((coordinates[i][0]-(win_x/2.)),(coordinates[i][1]-(win_y/2.))) for i in range(0,len(coordinates))]

    #get full length normalized coordinates for the positive and same length distractor
    #function also returns the updated normalized set
    #calling with 2 extra points so that those can be deleted later
    coordinates_2, coordinates_3, normalized = get_full_length_coordinates(nPoints, nTimes*3, normalized, path_length)
    #print(normalized)
    # print(len(coordinates_2))
    # print(len(coordinates_3))


    #using third length distractor function for full paths as well
    #coordinates_2,normalized = get_third_length_distractor_coordinates()

    #making distractors of length 50 (=150/3) on the fly
    #coordinates_4, coordinates_5, normalized = get_full_length_coordinates(nPoints, nTimes, normalized)
    #coordinates_4=coordinates_4[:50]
    #coordinates_5=coordinates_5[:50]

    # generate a new set of normalized coordinates beforehand, and pass it to the function to generate coordinates shifted by that much
    # normalized=np.random.choice(np.arange(-13,13),args.num_distractors+args.extra_dist, replace=False) #DISPLACE THE COORDINATES BY A RANDOM NUMBER OF PIXELS
    # normalized=normalized.astype(np.int8) #UNCOMMENT THIS LINE ALSO IF UNCOMMENTING THE ABOVE ONE TO SAVE MEMORY AND SPACE IN REPRESENTATION
    normalized=[0]*(args.num_distractors+args.extra_dist) #DO NOT DISPLACE. CONSTRAIN THE COORDINATES TO BE IN THE DEFINED VISIBLE VISUAL SCREEN

    coordinates_d, normalized = get_third_length_distractor_coordinates(nPoints, nTimes*3, normalized, num_distractors, path_length)

    # adding 4 extra distractors
    # 2 at the begining and end of the sequence
    extra_dist=args.extra_dist
    coordinates_e, normalized = get_third_length_distractor_coordinates(nPoints, nTimes*3, normalized, extra_dist, path_length)


    '''
    #updating nTimes here to counter
    #for the change in coordinate positions
    '''
    # nTimes=nTimes*3




    ##########################################################################################
    ##                  Start drawing here
    ##                  path_length is the same as number of frames
    ##########################################################################################


    images=np.zeros((path_length,32,32,channels), dtype=np.uint8)

    for frame in range(0,path_length):
        # import pdb; pdb.set_trace()
        for d in coordinates_d:
            rr, cc = rectangle(d[frame][0],d[frame][1],square_height,square_width)
            images[frame,rr, cc] = (0,255,0)
        
        for e in coordinates_e:
            rr, cc = rectangle(e[frame][0],e[frame][1],square_height,square_width)
            images[frame,rr, cc] = (0,255,0)
        
        #positive instance 


        # draw the two blobs
        # based on positive/negative sample
        rr, cc = draw_blobs(coordinates_2[0],blob_height,blob_width)
        images[frame,rr, cc] = (255,0,0) #128 #red starting open square
        # draw the end of path from coordinates_3 if NEGATIVE_SAMPLE, else from cordinates_2 if positive sample
        if args.NEGATIVE_SAMPLE:
            rr, cc = draw_blobs(coordinates_3[path_length-1],blob_height,blob_width)
        else:
            rr, cc = draw_blobs(coordinates_2[path_length-1],blob_height,blob_width)
        # rr, cc = rectangle_perimeter(np.floor(coordinates_2[63][0]), np.floor(coordinates_2[63][1]), 3, 3)
        images[frame,rr, cc] = (0, 0, 255) #light blue # (255, 184, 82) #Pastel Orange #(0,0,255) #blue ending open square




        # draw the points for two full length paths
        # since the full length coordinates are already floored, the blob drawn is floored as well, so squares have to be shifted by 1 pixel to fit in the center of the blob
        rr, cc = rectangle(coordinates_2[frame][0]+1,coordinates_2[frame][1]+1,square_height,square_width)
        images[frame, rr, cc] = (0,255,0)

        rr, cc = rectangle(coordinates_3[frame][0]+1,coordinates_3[frame][1]+1,square_height,square_width)
        images[frame, rr, cc] = (0,255,0)

        # try:
        #     os.makedirs(args.path+"/sample_"+str(sample))
        # except OSError:
        #     continue
        if args.save_image:
            imageio.imwrite(path_to_save+"/frame_"+str(frame)+".png", images[frame])
        
    # images_all[sample]=images
    # write every sample to disk as npy array
    np.save(path+"/"+str(int(not args.NEGATIVE_SAMPLE))+"_sample_"+str(sample), images, allow_pickle=False)

    # end=time.time()
    # print(end-start)

# writing to disk time
# could convert to sparse matrix before writing and to_dense after reading
# start=time.time()
# np.savez_compressed(path+"/samples_"+str(int(not args.NEGATIVE_SAMPLE))+"_"+str(args.start_sample)+"_"+str(args.num_samples)+".npz",images_all)
# end=time.time()
# print("Disk save: "+str(end-start))


# start=time.time()
# a=np.load(path+"/samples_"+str(int(not args.NEGATIVE_SAMPLE))+"_"+str(args.start_sample)+"_"+str(args.num_samples)+".npz")
# end=time.time()
# print("Disk load: "+str(end-start))

# print(False in (a['arr_0']==images_all))





# sys.exit(0)