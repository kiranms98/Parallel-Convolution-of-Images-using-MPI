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
image = Image.open("./space.jpg")               
data = np.asarray(image)

#any kernel/filter can be specified here to implement different transformations on the image
filter1 = np.asarray([[[1,0,0],[1,1,0],[0,0,1]]])
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

#calculate the number of rows assigned to each process
height = int(row/procs)
#if height is not perfectly divisible, the remainder is assigned to the last rank 'proc-1'
extra = int(row%procs)

#allocation of matrix segments to the various ranks
if(rank<procs-1):
    rank_data = a[height*rank:(rank+1)*height,:,:]
elif(rank==procs-1):
    rank_data = a[height*rank:(rank+1)*height+extra,:,:]

#START OF MESSAGE PASSING    
    
#send the last 'pad' number of rows to the top (ranks other than rank-'proc-1')
if(rank<procs-1):
    row_extra_bottom = a[(rank+1)*height-pad:(rank+1)*height,:,:]
    comm.send(row_extra_bottom,dest=rank+1)
    
#send the first 'pad' number of rows to the bottom (ranks other than rank-'0')
if(rank>0):
    row_extra_top = a[rank*height:rank*height+pad,:,:]
    comm.send(row_extra_top,dest=rank-1) 

#receive from the bottom
if (rank>0):
    row_extra_bottom = comm.recv(source=rank-1)
    #concatenate the extra rows to the bottom of rank's current matrix segment
    rank_data=np.concatenate((rank_data,row_extra_bottom),axis=0)

#receive from the top
if (rank<procs-1):
    row_extra_top = comm.recv(source=rank+1)
    #concatenate the extra rows to the top of rank's current matrix segment
    rank_data=np.concatenate((row_extra_top,rank_data),axis=0)   

#END OF MESSAGE PASSING   
    
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
    count = 0
    
    #data2 has the same dimensions as the input image
    data2 = np.zeros((data.shape[0],data.shape[1],data.shape[2]))
    
    #supposing rank 0 has 220 rows of transformed data, these rows are placed in rows 0-219 of data2
    #count is set to 220, so rank 1 places its data from row 220 onwards
    for i in range(len(gather)):
        data2[count:count+gather[i].shape[0],:,:]= gather[i]
        count = count + gather[i].shape[0] 
    
    #arrays can be transformed into images only if data type is uint8
    data2=data2.astype(np.uint8)
    Image.fromarray(data2).save("space.jpg")

#MPI Exit
MPI.Finalize()

#stop the clock and print the runtime of the code
later = time.time()
print("Time",later-now)
