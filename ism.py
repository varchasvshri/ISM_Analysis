from collections import defaultdict 
import numpy as np
import matplotlib.pyplot as plt

#Class to represent a graph 
class Graph:  
    def __init__(self, vertices): 
        self.V = vertices 
  
    # A utility function to print the solution 
    def printSolution(self, reach): 
        print ("Following matrix transitive closure of the given graph ")     
        for i in range(self.V): 
            for j in range(self.V): 
                print (reach[i][j],end=" ")
            print("\n")
      
      
    # Prints transitive closure of graph[][] using Floyd Warshall algorithm 
    def transitiveClosure(self,graph): 
        reach =[i[:] for i in graph] 
        for k in range(self.V): 
            for i in range(self.V): 
                for j in range(self.V): 
                    reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j]) 
        return reach 

def print_final_reachability(initial, final):
    # print(initial)
    for i in range(n): 
            for j in range(n): 
                if(final[i][j]==1 and initial[i][j]==0):
                    print('1*',end=" ")
                elif(final[i][j]==1):
                    print('1',end=" ")
                else:
                    print('0',end=" ")
            print("\n")

def Level_Partioning(final):
    common_mat = []
    for i in range(n):
        temp_reach = []
        temp_antec = []
        for j in range(n):
            if(final[i][j]==1):
                temp_reach.append(j)
            if(final[j][i]==1):
                temp_antec.append(j)
        common_mat.append(temp_reach)
        common_mat.append(temp_antec)
    return common_mat

def stop_crit(level):
    for x in level:
        if(x==0):
            return True
    return False        

def xandy(final):
    Driving_power = []
    Dependence_power = []

    for i in range(n):
        countx=0
        county=0
        for j in range(n):
            if(final[i][j]==1):
                countx = countx + 1
            if(final[j][i]==1):
                county = county + 1
        Driving_power.append(countx)
        Dependence_power.append(county)
    return Driving_power, Dependence_power

def find_level(intersection_set, common_mat):
    levels = np.zeros(n, dtype=int)
    count = 1

    while(stop_crit(levels)):
        store = []
        for i in range(n):
            if(len(intersection_set[i])!=0 and set(intersection_set[i]) == set(common_mat[2*i])):
                levels[i] = count
                store.append(i)
        count = count + 1
        for x in store:
            for i in common_mat:
                if x in i: i.remove(x)
            for i in intersection_set:
                if x in i: i.remove(x)
    return levels

def plot_it(Driving_power, Dependence_power):
    plt.scatter(Dependence_power, Driving_power)
    pts = dict() #pts is dictionary mapping from tuple of points to list of index corresponding to that
    for i, txt in enumerate(range(n)):
    	t = (Dependence_power[i], Driving_power[i]) #t is placeholder variable for coordinate
    	if t in pts:
    		pts[t].append(txt+1)
    	else:
    		pts[t]=[txt+1]

    for i, txt in enumerate(range(n)):
       	t = (Dependence_power[i], Driving_power[i])
        plt.annotate(pts[t],t)


    x1, y1 = [-1, n+1], [n/2, n/2]
    x2, y2 = [n/2, n/2], [-1, n+1]
    plt.plot(x1, y1, x2, y2)
    
    plt.xlim(0,n+1)
    plt.ylim(0,n+1)
    plt.xlabel('Dependence') 
    plt.ylabel('Driving Power') 
    plt.title('Micmac Analysis')
    plt.show()    

n = int(input('Dimension of your Initial Reachability matrix : '))
area = input('name of input file : ')

graph = np.loadtxt(area, usecols=range(n))

g= Graph(n) 
final = g.transitiveClosure(graph)
temp = np.loadtxt(area, usecols=range(n))
print_final_reachability(temp, final)
Driving_power, Dependence_power = xandy(final)

common_mat = Level_Partioning(final)

intersection_set = []
for i in range(n):
    intersection_set.append(list(set(common_mat[2*i]) & set(common_mat[2*i + 1])))

levels = find_level(intersection_set, common_mat)

for i in range(n):
    print('Level in TISM for E%d is %d'%(i+1,levels[i]))

plot_it(Driving_power, Dependence_power)
