// task1
// 使用基于种子标记的分水岭算法（OpenCV自带watershed）对输入图像进行过分割
// 用户输入图像和整数K，要求程序自动计算K个随机种子点，确保各种子点之间的距离均 > (M*N/K)0.5（参考泊松圆盘采样+贪心策略）
// 然后让程序在原图中标出各种子点的位置及编号，并采用半透明+随机着色的方式给出分水岭算法的可视化结果。

// solution
// adopt general framework of cv2-watershed.cpp
// add features:
// input img path (with size of MxN) and int k,
// compute random seeds of number k, while keeping min distance of seeds larger than (M*N/K)0.5)
// display image covered with seeds, print seed order num and coordinates
// compute boundaries using watershed?
// refer to cv2-watershed.cpp and put on random colors...
// visualize result