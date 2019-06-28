import java.util.Map;
import java.util.TreeMap;

class MyCalendarTwo {

  private TreeMap<Integer, Integer> pointMap;

  public MyCalendarTwo() {
    pointMap = new TreeMap<>();
  }

  public boolean book(int start, int end) {
    pointMap.put(start, pointMap.getOrDefault(start, 0) + 1);
    pointMap.put(end, pointMap.getOrDefault(end, 0) - 1);
    int maxCount = 0, count = 0;
    for (Map.Entry<Integer, Integer> entry : pointMap.entrySet()) {
      count += entry.getValue();
      maxCount = Math.max(maxCount, count);
      if (maxCount > 2) {
        pointMap.put(start, pointMap.get(start) - 1);
        if (pointMap.get(start) == 0) {
          pointMap.remove(start);
        }
        pointMap.put(end, pointMap.get(start) + 1);
        if (pointMap.get(end) == 0) {
          pointMap.remove(end);
        }
        return false;
      }
    }
    return true;
  }
}