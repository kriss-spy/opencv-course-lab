// recursive dfs
// not suitable for task2

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * @author LiYang
 * @ClassName FourColorTheorem
 * @Description 四色定理探索实践类
 * @date 2019/12/7 22:32
 */
public class FourColorTheorem {

    // 根据四色定理，最多只使用四种颜色，就可以为任何地图
    // 着色，并且保证相邻接壤的区域必须用不同的颜色
    // ---------------------------------------------
    // 注意，如果你的地图足够特别，也是有可能三种，甚至更少
    // 的颜色种类就完成着色，此时可以将下面的数字改小，然后
    // 查看控制台是否有结果输出，有结果输出则表示可行
    private static final int MIN_NECESSARY_MAP_COLOR = 4;

    /**
     * 返回示例地图接壤信息的邻接矩阵
     * 
     * @return 示例地图接壤信息的邻接矩阵
     */
    public static int[][] initMapMatrix() {
        // 直接返回记录示例地图接壤信息的邻接矩阵
        // 要运行自己的地图，可以修改此邻接矩阵
        return new int[][] {
                { 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 },
                { 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
                { 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 },
                { 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0 },
                { 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0 },
                { 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1 },
                { 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0 },
                { 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0 },
                { 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1 },
                { 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1 },
                { 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0 }
        };
    }

    /**
     * 深拷贝数组
     * 
     * @param source 数组拷贝源
     * @return 数组副本
     */
    public static int[] copyArray(int[] source) {
        // 创建新的数组
        int[] copy = new int[source.length];

        // 拷贝赋值新数组
        System.arraycopy(source, 0, copy, 0, source.length);

        // 返回深拷贝的副本
        return copy;
    }

    /**
     * 地图区域着色算法的递归执行方法
     * 
     * @param nextArea  着色区域
     * @param color     着色使用的颜色
     * @param colorPlan 着色计划记录数组
     * @param matrix    区域接壤的邻接矩阵
     */
    public static void coloringMap(int nextArea, int color, int[] colorPlan, int[][] matrix) {
        // 将当前区域着色
        colorPlan[nextArea] = color;

        // 如果已经全部着色，则打印结果
        if (nextArea == colorPlan.length - 1) {

            // 打印当前递归分支的着色方案
            System.out.println("找到着色方案：" + Arrays.toString(colorPlan));

            // 结束当前递归分支
            return;
        }

        // 当前区域更新为下一个区域
        // 准备为下一个区域进行着色
        nextArea++;

        // 下一个区域的可用颜色集
        Set<Integer> availableColor = new HashSet<>();

        // 初始化可用颜色集
        for (int i = 1; i <= MIN_NECESSARY_MAP_COLOR; i++) {

            // 先全部加入，再用排除法去掉周边接壤颜色
            availableColor.add(i);
        }

        // 遍历邻接矩阵，找到下一个区域的所有接壤区域
        for (int i = 0; i < matrix.length; i++) {

            // 如果当前区域接壤，且已经着色
            if (matrix[nextArea][i] > 0 && colorPlan[i] > 0) {

                // 将接壤的已用颜色剔除
                availableColor.remove(colorPlan[i]);
            }
        }

        // 遍历下一个区域的所有可用颜色
        for (int available : availableColor) {

            // 分别递归调用本算法，尝试下一个区域着所有可用颜色集
            coloringMap(nextArea, available, copyArray(colorPlan), matrix);
        }
    }

    /**
     * 地图区域着色算法的驱动方法
     * 
     * @param matrix 地图接壤的邻接矩阵
     */
    public static void coloringMap(int[][] matrix) {
        // 我们从0号区域开始，从哪个区域开始都一样
        int nextArea = 0;

        // 颜色代号是1-4，本质是等价可互换的，用1开始就行
        int color = 1;

        // 初始化着色方案
        int[] colorPlan = new int[matrix.length];

        // 调用地图区域着色算法的递归执行方法
        coloringMap(nextArea, color, colorPlan, matrix);
    }

    /**
     * 运用四色定理，运行示例地图着色的算法
     * 
     * @param args
     */
    public static void main(String[] args) {
        // 获得记录示例地图接壤关系的邻接矩阵
        int[][] matrix = initMapMatrix();

        // 运行示例地图着色的算法，并从控制台
        // 查看打印的着色方案
        coloringMap(matrix);
    }

}
