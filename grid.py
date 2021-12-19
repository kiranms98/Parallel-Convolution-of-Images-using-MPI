from IPython import display
from PIL import Image
import numpy as np
import mpi4py as mp
import time

#start the clock
now = time.time()

#disable auto initialize and finalize
mp.rc.initialize = False
mp.rc.finalize = False
from mpi4py import MPI

#import the image file usin PIL Library
image = Image.open("./lion.jpg")
data = np.asarray(image)

#any kernel/filter can be specified here to implement different transformations on the image
filter1 = np.asarray([[[1,0,0,1,1,0,0],[1,0,0,1,0,1,0],[0,0,1,1,0,0,1],[1,0,1,0,1,0,1],[1,0,0,1,0,1,0],[1,0,0,1,1,0,0],[0,0,1,1,0,0,1]]])
filter1 = np.reshape(filter1,(filter1.shape[1],filter1.shape[2],filter1.shape[0]))
size = filter1.shape[0]

row = data.shape[0]
column = data.shape[1]
pad = int((size-1)/2)

#addtional zeros are added across the border of the image matrix 
#so that the transformed image has the same dimensions as input image
a = np.pad(data,((pad,pad),(pad,pad),(0,0)))

#initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
procs = comm.size

#this code works specifically when the number of specified processes is a perfect square
#for every rank-number, 'rank_div' and 'rank_mod' are calculated which help in locating
#and allocating a rank's segment

#Below is an example for 9 processes. Rank_div=rank/3 and rank_mod=rank%3

#   Ranks      Rank_div     Rank_mod
# [6  7  8     [2  2  2     [0  1  2
#  3  4  5      1  1  1      0  1  2
#  0  1  2]     0  0  0]     0  1  2]

#in all comments below, the above location matrix should be referred to for clear comprehension
#eg: rightmost ranks refers to ranks-2,5,8; topmost ranks - 6,7,8

root_procs=int(np.sqrt(procs))
rank_div = int(rank/root_procs)
rank_mod = int(rank%root_procs)

#calculate the height of each segment
height = int(row/root_procs)

#if height is not perfectly divisible, every rank with max(rank_div) is allocated the extra rows
extra_h = int(row%root_procs)

#calculate the width of each segment
width = int(column/root_procs)

#if width is not perfectly divisible, every rank with max(rank_mod) is allocated the extra columns
extra_w = int(column%root_procs)

################# ALLOCATION OF MATRIX SEGMENTS TO EACH RANK ###################

#all ranks other than the topmost ranks
if(rank_div<root_procs-1):
    
    #all ranks other than the rightmost ranks
    if(rank_mod<root_procs-1):
        rank_data = a[height*rank_div:(rank_div+1)*height,width*rank_mod:(rank_mod+1)*width,:]
        
    #for the rightmost ranks
    elif(rank_mod==root_procs-1):
        rank_data = a[height*rank_div:(rank_div+1)*height,width*rank_mod:(rank_mod+1)*width+extra_w,:]   

#all the topmost ranks
elif(rank_div==root_procs-1):
    
    #all ranks other than the rightmost ranks
    if(rank_mod<root_procs-1):
        rank_data = a[height*rank_div:(rank_div+1)*height+extra_h,width*rank_mod:(rank_mod+1)*width,:]
        
    #for the rightmost ranks    
    elif(rank_mod==root_procs-1):
        rank_data = a[height*rank_div:(rank_div+1)*height+extra_h,width*rank_mod:(rank_mod+1)*width+extra_w,:]

        
################# END OF MATRIX SEGMENT ALLOCATION ###################       
        
    
###########START OF TOP AND BOTTOM SENDS AND RECEIVES#################

#send the last 'pad' number of rows to the top (ranks other than topmost ranks)
if(rank_div<root_procs-1):
    
    #for the ranks other than the rightmost ranks
    if(rank_mod<root_procs-1):
        row_extra_bottom = a[(rank_div+1)*height-pad:(rank_div+1)*height,width*rank_mod:(rank_mod+1)*width,:]
        comm.send(row_extra_bottom,dest=rank+root_procs)
        
    #for the rightmost ranks
    elif(rank_mod==root_procs-1):
        row_extra_bottom = a[(rank_div+1)*height-pad:(rank_div+1)*height,width*rank_mod:(rank_mod+1)*width+extra_w,:]
        comm.send(row_extra_bottom,dest=rank+root_procs)
        
#send the first 'pad' number of rows to the bottom (ranks other than bottommost ranks)
if(rank_div>0):
    
    #for the ranks other than the rightmost ranks
    if(rank_mod<root_procs-1):
        row_extra_top = a[rank_div*height:rank_div*height+pad,width*rank_mod:(rank_mod+1)*width,:]
        comm.send(row_extra_top,dest=rank-root_procs)
     
    #for the rightmost ranks
    elif(rank_mod==root_procs-1):
        row_extra_top = a[rank_div*height:rank_div*height+pad,width*rank_mod:(rank_mod+1)*width+extra_w,:]
        comm.send(row_extra_top,dest=rank-root_procs)

#receive from the bottom
if (rank_div>0):
    row_extra_bottom = comm.recv(source=rank-root_procs)
    
    #concatenate the extra rows to the bottom of rank's current matrix segment
    rank_data=np.concatenate((rank_data,row_extra_bottom),axis=0) 
    
#receive from the top
if (rank_div<root_procs-1):
    row_extra_top = comm.recv(source=rank+root_procs)
    
    #concatenate the extra rows to the top of rank's current matrix segment
    rank_data = np.concatenate((row_extra_top,rank_data),axis=0)

    
###########END OF TOP AND BOTTOM SENDS AND RECEIVES#################

###########START OF LEFT AND RIGHT SENDS AND RECEIVES###############

#this segment has to be dealt with differently since we will now we concatenating the
#columns to updated 'rank_data' which presently has extra rows. The message sending
#must take this into account and send columns of appropriate height

#send the last 'pad' number of columns to the right (ranks other than rightmost ranks)
if(rank_mod<root_procs-1):
    
    #for the bottommost ranks which now have 'pad' number of extra rows in rank_data
    if(rank_div==0):
        row_extra_right = a[0:height+pad,width*(rank_mod+1)-pad:(rank_mod+1)*width,:]
        comm.send(row_extra_right,dest=rank+1)
        
    #for the topmost ranks which now have 'pad+extra_h' number of extra row in rank_data
    elif(rank_div==root_procs-1):
        row_extra_right = a[rank_div*height-pad:(rank_div+1)*height+extra_h,width*(rank_mod+1)-pad:(rank_mod+1)*width,:]
        comm.send(row_extra_right,dest=rank+1)
        
    #for the rest of the ranks which now have '2*pad' number of extra row in rank_data
    else:
        row_extra_right = a[rank_div*height-pad:(rank_div+1)*height+pad,width*(rank_mod+1)-pad:(rank_mod+1)*width,:]
        comm.send(row_extra_right,dest=rank+1)

#send the first 'pad' number of columns to the left (ranks other than leftmost ranks)
if(rank_mod>0):
    
    #for the bottommost ranks which now have 'pad' number of extra rows in rank_data
    if(rank_div==0):
        row_extra_left = a[0:height+pad,width*(rank_mod):(rank_mod)*width+pad,:]
        comm.send(row_extra_left,dest=rank-1)
        
    #for the topmost ranks which now have 'pad+extra_h' number of extra row in rank_data
    elif(rank_div==root_procs-1):
        row_extra_left = a[rank_div*height-pad:(rank_div+1)*height+extra_h,width*(rank_mod):(rank_mod)*width+pad,:]
        comm.send(row_extra_left,dest=rank-1)
        
    #for the rest of the ranks which now have '2*pad' number of extra row in rank_data
    else:
        row_extra_left = a[rank_div*height-pad:(rank_div+1)*height+pad,width*(rank_mod):(rank_mod)*width+pad,:]
        comm.send(row_extra_left,dest=rank-1)        
        
#receive from the left
if (rank_mod>0):
    row_extra_left = comm.recv(source=rank-1)
    
    #concatenate the extra rows to the left of rank's updated matrix segment
    rank_data=np.concatenate((rank_data,row_extra_left),axis=1)

#receive from the right
if (rank_mod<root_procs-1):
    row_extra_right = comm.recv(source=rank+1)
    
    #concatenate the extra rows to the right of rank's updated matrix segment
    rank_data=np.concatenate((row_extra_right,rank_data),axis=1)

    
###########END OF LEFT AND RIGHT SENDS AND RECEIVES#################


#each rank initializes a matrix 'final_data' which stores the results of the convolution
#dimensions of 'final_data' are lesser than rank_data(which is zero padded). 'final_data' 
#has the same width as the original unpadded input matrix
final_data=np.zeros((rank_data.shape[0]-pad*2,rank_data.shape[1]-pad*2,rank_data.shape[2]))

#counter variables help store the result of convolution in the appropriate location of 
#'final_data'. c2 is set to 0 after every horizontal pass, c1 is set to zero after the
#entire channel has been traversed
c1=0
c2=0
c3=0

#k loops over the image channels, i loops over rows, j loops over columns
for k in range(rank_data.shape[2]):
    c1=0
    for i in range(rank_data.shape[0]+1-size):
        c2=0
        for j in range(rank_data.shape[1]+1-size):
            #convolution operation by elementwise multiplication,summation
            final_data[c1,c2,c3] = np.sum(np.multiply(rank_data[i:i+size,j:j+size,k],filter1))
            c2=c2+1
        c1=c1+1
    c3=c3+1

#all the segments of transformed data are gathered and stored in 'gather'
gather = comm.gather(final_data,root=0)

#to ensure that every process has stored its 'final_data' in gather before
#rank 0 assembles the final transformed image
comm.barrier() 

if rank==0:
    count1 = 0
    
    #data2 has the same dimensions as the input image
    data2 = np.zeros((data.shape[0],data.shape[1],data.shape[2]))
       
    #the inner loop arranges the segments horizontally, outer loop moves from row to row
    #every segment can be accurately located using i*root_procs+j. Here i represents rank_div while
    #j represents rank_mod
 
    for i in range(root_procs):
        
        #supposing rank 0 has 220 columns of transformed data, these columns are placed in columns 0-219 of data2
        #count2 is set to 220, so rank 1 places its data from columns 220 onwards
        count2 = 0
        
        for j in range(root_procs):
            data2[count1:count1+gather[i*root_procs+j].shape[0],count2:count2+gather[i*root_procs+j].shape[1],:]= gather[i*root_procs+j]
            count2 = count2 + gather[i*root_procs+j].shape[1]
            
        #count1 is updated so the next set of segments are placed above the newly placed segments
        count1 = count1 + gather[i*root_procs].shape[0]
     
    #arrays can be transformed into images only if data type is uint8
    data2=data2.astype(np.uint8)
    Image.fromarray(data2).save("gfg3.png")


#MPI Exit
MPI.Finalize()

#stop the clock and print the runtime of the code
later = time.time()
print("Time",later-now)