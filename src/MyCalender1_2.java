import java.util.Map;
import java.util.TreeMap;

class MyCalender1_2 {

  private TreeMap<Integer, Integer> points;

  public MyCalender1_2() {
    points = new TreeMap<>();
  }

  public boolean book(int start, int end) {
    Map.Entry<Integer, Integer> floorEntry = points.floorEntry(start);
    if (floorEntry != null && start < floorEntry.getValue()) return false;
    Map.Entry<Integer, Integer> ceilingEntry = points.ceilingEntry(end);
    if (ceilingEntry != null && end > ceilingEntry.getKey()) return false;
    points.put(start, end);
    return true;
  }
}