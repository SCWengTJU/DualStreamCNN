import numpy as np
import os

class transform:
    def __init__(self, dir):
        self.dir = dir
    pass
    def intToString(self, num):
        one = num % 1000 / 100
        two = num % 100 / 10
        three = num % 10
        return str(one) + str(two) + str(three)
    pass
    def norm(self,fileName,name):
        floatNum = np.loadtxt(fileName)
        size_3 = floatNum.size / 80  
        floatNum = floatNum.reshape(size_3, 20, 4) 
        newfloatNum=np.zeros(np.array([68,20,4]))
        for i in range(68):
            relative_position=(i/68.0*size_3) 
            becut=int(np.floor(relative_position))
            left=relative_position-becut
            over=1-left
            if becut==size_3-1:
                newfloatNum[i,:,:]=floatNum[becut,:,:]
            else:
                newfloatNum[i, :, :]=floatNum[becut,:,:]*left+floatNum[becut+1,:,:]*over
        ans=np.zeros(np.array([68*20,4]))
        row=0
        for i in range(68):
            for j in range(20):
                for k in range(4):
                    ans[row][k]=newfloatNum[i][j][k]
                row=row+1
        if not os.path.exists('./test_67'):
            os.mkdir('./test_67')
        np.savetxt('./test_67/'+name,ans)
    pass
    def getCheckFile(self):
        for sss in range(150):
            for nnn in range(50):
                name = 'a' + str(sss) + '_r' + str(nnn)
                if os.path.exists(self.dir + name) == False:
                    continue
                print name
                self.norm(self.dir + name, name)
    pass



pass

trans=transform('./test_raw/')
print '~'
trans.getCheckFile()