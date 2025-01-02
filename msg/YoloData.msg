string[] classes #class name of each rectangle
float32[] rectangles 
float64 time

#every 4 elements of rectangles corresponds to the 4 vertices of a single box
#y min1, xmin1 ymax1 xmax1 ymin2 xmin2 ymax2 xmax2 ...
# [0]     [1]   [2]   [3]   [4]   [5]   [6]   [7]  ...
#x and y values may be outside of screen ( ie negative or  bigger then number of rows or columns)
# to fix simply set each to the max of 
