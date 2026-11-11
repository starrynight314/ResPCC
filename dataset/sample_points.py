# https://github.com/viscom-ulm/MCCNN/blob/master/utils/DataSet.py
import torch
import numpy as np
import time
from sklearn.neighbors import KDTree
#点云采样的工具类，实现了多种不同的点云采样策略




class SampleMethod():
    def __init__(self, rate=0.8):
        self.rate = rate #采样比例
        self.randomState_ = np.random.RandomState(int(time.time())) #这是NumPy的随机数生成器类创建一个具有固定种子的随机数生成器实例；相同的种子会产生相同的随机数序列

    #分割采样 偏向选择点云某个方向的点 对于最大范围的轴的后半部分的点一定选择
    def _non_uniform_sampling_split_(self,  points, inNumPoints, inFeatures=None, numPoints=0, lowProbability=0.1):
        '''
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list.
            inFeatures (nxm np.array): List of point features.
            numPoints (int): Number of points to sample. If 0, all the points are selected.对所有点都进行一次随机选择采样
            lowProbability (float): Probability used for the points in the second half of the bounding box.
            :return:
        '''
        shuffle_idx = np.random.choice(points.shape[0], points.shape[0], replace=False) #从0-points.shape[0]之间随机选择points.shape[0]个随机数，并且不重复
        points = points[shuffle_idx, :] #根据shuffle_idx的索引，重新排列points中的点
        coordMax = np.max(points, axis=0) #计算每个轴上的坐标的最大值
        coordMin = np.min(points, axis=0) #计算每个轴上的坐标的最小值
        aabbSize = coordMax - coordMin #计算每个轴上的坐标的范围

        largestAxis = np.argmax(aabbSize) #返回坐标范围最大的那个轴的索引
        auxOutPts = []
        auxOutInFeatures = []
        exitVar = False #退出循环的标志
        numAddedPts = 0 #number of added points的缩写 记录已经添加的点数
        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            for i in range(inNumPoints):
                currPt = points[i]
                ptPos = (currPt[largestAxis] - coordMin[largestAxis]) / aabbSize[largestAxis] #计算当前点在最大轴上的相对归一化位置
                if ptPos > 0.5:
                    probVal = 1.0 #如果当前点在最大轴上的相对归一化位置大于0.5，则概率为1.0
                    rndNum = 0 #随机数为0
                else:
                    probVal = lowProbability
                    rndNum = self.randomState_.random_sample() #利用随机数生成器生成一个0-1之间的随机数
                # Determine if we select the point.

                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(currPt) #将当前点添加到输出缓冲区 所以对于不在后半段的点 只有0.1的概率被选中
                    if not (inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])
                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints): #numPoints目标采样数目
                        exitVar = True
                        break
            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        if not (inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures],axis=1)

        return npOutPts


    #基于朗伯光照模型的非均匀采样方法
    def _non_uniform_sampling_lambert_(self, viewDir, points, normals, inNumPoints, inFeatures=None,
         numPoints=0):
        '''
        viewDir (3 np.array): View vector used to compute the probability of each point. 观察方向向量 根据观察者的视线方向计算每个点的概率 朗伯采样基于朗伯光照模型，这是一种计算机图形学中的光照模型；表面反射的光强度与观察角度和表面法向量的夹角余弦成正比
        points (nx3 np.array): List of points.
        normals (nx3 np.array): List of point normals.
        inNumPoints (int): Number of points in the list.
        inFeatures (nxm np.array): List of point features.
        numPoints (int): Number of points to sample. If 0, all the points are selected.
        :return:
        '''

        auxOutPts = []
        auxOutInFeatures = []

        exitVar = False
        numAddedPts = 0
        while not exitVar:
            for i in range(inNumPoints):
                # Compute the point probability.
                probVal = np.dot(viewDir, normals[i]) #计算观察方向向量和点法向量的点积 点积越大，说明点越面向观察者 所以概率越大
                probVal = pow(np.clip(probVal, 0.0, 1.0), 0.5) #将点积归一化到0-1之间，然后取平方根
                # Determine if we select the point.
                rndNum = self.randomState_.random_sample()
                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(points[i])
                    if not (inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])
                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break
            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        if not (inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures], axis=1)

        return npOutPts

    #遮挡采样（Occlusion Sampling）
    def _non_uniform_sampling_occlusion_(self, viewDir, points, normals, inNumPoints, inFeatures=None,
                                          numPoints=0, screenResolution=128):
        '''
        viewDir (3 np.array): View vector used to compute the visibility of each point.
        points (nx3 np.array): List of points.
        normals (nx3 np.array): List of point normals.
        inNumPoints (int): Number of points in the list.
        inFeatures (nxm np.array): List of point features.
        inLabels (nxl np.array): List of point labels.
        numPoints (int): Number of points to sample. If 0, all the points are selected.
        :return:
        '''
        xVec = np.cross(viewDir, np.array([0.0, 1.0, 0.0])) #得到x轴方向向量
        xVec = xVec / np.linalg.norm(xVec) #归一化x轴方向向量
        yVec = np.cross(xVec, viewDir) #计算x轴方向向量和观察方向向量的叉积 得到y轴方向向量
        yVec = yVec / np.linalg.norm(yVec) #归一化y轴方向向量

        # Compute the bounding box.
        coordMax = np.max(points, axis=0)
        coordMin = np.min(points, axis=0)
        diagonal = np.linalg.norm(coordMax - coordMin) * 0.5 #计算点云的包围盒对角线长度的一半 np.linalg.norm 计算向量的模长
        center = (coordMax + coordMin) * 0.5 #计算点云的包围盒中心点

        # Create the screen pixels
        screenSize = screenResolution
        pixelSize = diagonal / (float(screenSize) * 0.5) #以确保不管从哪个角度屏幕都可以覆盖整个点云 所以screen长宽是最长边的一半
        screenPos = center - viewDir * diagonal - xVec * diagonal - yVec * diagonal #屏幕左下角在空间中的位置
        screenZBuff = np.full([screenSize, screenSize], -1.0) #初始化屏幕深度缓冲区 所有像素的深度值都为-1
        # Compute the z value and pixel id in which each point is projected.
        pixelIds = [[-1, -1] for i in range(inNumPoints)] #初始化点投影到屏幕的像素ID 所有像素的ID都为-1
        zVals = [1.0 for i in range(inNumPoints)] #初始化点投影到屏幕的深度值 所有深度值都为1.0
        for i in range(inNumPoints):
            # If the point is facing the camera.
            if np.dot(normals[i], viewDir) < 0.0: #面向投影者的点
                # Compute the z value of the pixel.
                transPt = points[i] - screenPos #转化为相对于屏幕左下角的世界坐标
                transPt = np.array([   #将世界坐标转化为屏幕坐标
                    np.dot(transPt, xVec),
                    np.dot(transPt, yVec),
                    np.dot(transPt, viewDir) / (diagonal * 2.0)]) #z缓冲区的需要，归一化深度值
                zVals[i] = transPt[2] #更新点投影到屏幕的深度值

                # Compute the pixel id in which the point is projected.
                transPt = transPt / pixelSize
                pixelIds[i][0] = int(np.floor(transPt[0]))
                pixelIds[i][1] = int(np.floor(transPt[1]))

                # Update the z-buffer.
                if screenZBuff[pixelIds[i][0]][pixelIds[i][1]] > zVals[i] or screenZBuff[pixelIds[i][0]][
                    pixelIds[i][1]] < 0.0:
                    screenZBuff[pixelIds[i][0]][pixelIds[i][1]] = zVals[i] #更新深度值为离投影屏面最近的点的深度值

        auxOutPts = []
        auxOutInFeatures = []
        exitVar = False
        numAddedPts = 0
        # Iterate over the points until we have the desired number of output points.
        while not exitVar:  #判断点是否被遮挡 如果点的深度值与Z缓冲区中的深度值接近，说明点可见 存储
            # Iterate over the points.
            for i in range(inNumPoints):
                # Determine if the point is occluded.
                if (zVals[i] - screenZBuff[pixelIds[i][0]][pixelIds[i][1]]) < 0.01:
                    # Store the point in the output buffers.
                    auxOutPts.append(points[i])
                    if not (inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])

                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break

            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        if not (inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures], axis=1)

        return npOutPts

    #偏向选择包围盒边缘部分的点 但是概率是连续的
    def _non_uniform_sampling_gradient_(self, points, inNumPoints, inFeatures=None, numPoints=0):
        """Method to non-uniformly sample a point cloud using the gradient protocol. The probability to select a
        point is based on its position alogn the largest axis of the bounding box.

        Args:
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list.
            inFeatures (nxm np.array): List of point features.
            inLabels (nxl np.array): List of point labels.
            numPoints (int): Number of points to sample. If 0, all the points are selected.

        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """
        # Compute the bounding box.
        shuffle_idx = np.random.choice(points.shape[0], points.shape[0], replace=False)
        points = points[shuffle_idx, :]
        coordMax = np.max(points, axis=0)
        coordMin = np.min(points, axis=0)
        aabbSize = coordMax - coordMin
        largestAxis = np.argmax(aabbSize)

        auxOutPts = []
        auxOutInFeatures = []
        exitVar = False
        numAddedPts = 0

        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            # Iterate over the points.
            for i in range(inNumPoints):
                # Compute the point probability.
                currPt = points[i]
                probVal = (currPt[largestAxis]-coordMin[largestAxis]-aabbSize[largestAxis]*0.2  #计算点在最大轴上的位置 归一化到0-1之间
                    )/(aabbSize[largestAxis]*0.6)
                probVal = pow(np.clip(probVal, 0.01, 1.0), 1.0/2.0)
                # Determine if we select the point.
                rndNum = self.randomState_.random_sample()
                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(currPt)
                    if not(inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])

                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):

                        exitVar = True
                        break

            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        if not (inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures], axis=1)

        return npOutPts

    #保留所有的点
    def _uniform_sampling_(self, points, inFeatures=None, ):
        """Method to uniformly sample a point cloud.

        Args:
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list.
            inFeatures (nxm np.array): List of point features.
        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """

        npOutPts = np.array(points)
        if not (inFeatures is None):
            npOutInFeatures = np.array(inFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures], axis=1)

        return npOutPts

    #K近邻非均匀采样（KNN Non-uniform Sampling）
    def _knn_non_uniform_sampling_(self, points, inFeatures, min_Points=100):
        key_point = points[0, :].reshape(-1,3) #选择第一个点作为关键点 调整成三列的形式
        kdt = KDTree(points) #构建KD树 用于快速最近邻搜索
        keep_idx = kdt.query(key_point[:, :3], k=points.shape[0] // 2, return_distance=False)[0] #查询关键点，返回它附近的k个相邻点，只返回索引 选择key_point的所有行的前三列
        keep_points = points[keep_idx, :]
        keep_features = inFeatures[keep_idx,:]
        
        #剩余点
        need_sample_points = points[np.setdiff1d(np.arange(points.shape[0]), keep_idx), :] #选择不在关键点附近的点 np.setdiff1d计算两个数组的差集 是索引表示
        need_sample_points_features = inFeatures[np.setdiff1d(np.arange(points.shape[0]), keep_idx), :]

        if need_sample_points.shape[0] // 10 < min_Points - len(keep_idx):
            real_need_sample_point_num = min_Points - len(keep_idx)
        else:
            real_need_sample_point_num = need_sample_points.shape[0] // 10 #采的个数超过min_points

        sample_idx = np.random.choice(np.arange(need_sample_points.shape[0]), real_need_sample_point_num) #从剩下的点中随机采样

        reshample_point = need_sample_points[sample_idx, :]
        resample_features = need_sample_points_features[sample_idx, :]
        final_point = np.concatenate([keep_points, reshample_point], 0)
        final_features = np.concatenate([keep_features, resample_features], 0)

        return np.concatenate([final_point, final_features],1)


    def random_sample(self, points, attribute=None, min_point=100):
        # random_strategy = [self._non_uniform_sampling_lambert_, self._non_uniform_sampling_occlusion_,
        #                    self._non_uniform_sampling_split_, self._uniform_sampling_]
        choice = self.randomState_.choice(4,1)[0]

        # print(choice) 

        numPoints = int(self.rate * points.shape[0])
        if numPoints <= min_point:
            numPoints = min_point

        if choice==0:
            point_with_attribute = self._non_uniform_sampling_split_(points, len(points), attribute, numPoints=numPoints)
        elif choice==1:
            point_with_attribute = self._non_uniform_sampling_gradient_(points, len(points),attribute, numPoints=numPoints)
        elif choice==2:
            point_with_attribute = self._knn_non_uniform_sampling_(points, attribute, min_Points=min_point )
        #     auxView = (self.randomState_.rand(3) * 2.0) - 1.0
        #     auxView = auxView / np.linalg.norm(auxView)
        #     point_with_attribute = self._non_uniform_sampling_lambert_(auxView, points, inNumPoints=len(points),normals=attribute[:,:3],inFeatures=attribute, numPoints=numPoints)
        else:
            point_with_attribute = self._uniform_sampling_(points, attribute)

        if point_with_attribute.shape[0] < min_point:
            print(choice)
            print('original points shape:', points.shape)
            print('points with attribute shape:', point_with_attribute.shape)
            raise  ValueError
        assert  point_with_attribute.shape[0] >= min_point

        return point_with_attribute
