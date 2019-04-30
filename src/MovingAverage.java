import java.util.LinkedList;
import java.util.Queue;

class MovingAverage {

  private int curSum;
  private int totalSize;

  private Queue<Integer> q;

  /**
   * Initialize your data structure here.
   */
  public MovingAverage(int size) {
    totalSize = size;
    curSum = 0;
    q = new LinkedList<>();
  }

  public double next(int val) {
    if (q.size() == totalSize) {
      Integer toRemove = q.poll();
      curSum -= toRemove;
    }
    q.add(val);
    curSum += val;
    return (double) curSum / (double) q.size();
  }
}