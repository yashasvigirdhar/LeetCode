//import java.util.List;
//import java.util.Map;
//import java.util.TreeMap;
//
//public class SummaryRangesII {
//
//  /**
//   * Initialize your data structure here.
//   */
//
//  TreeMap<Integer, Integer> map;
//
//  public SummaryRangesII() {
//    map = new TreeMap<>();
//  }
//
//  public void addNum(int val) {
//    if (map.size() == 0) {
//      map.put(val, val);
//      return;
//    }
//    Integer l = map.lowerKey(val);
//    Integer h = map.higherKey(val);
//
//    if(l!= null && h!= null && val == map.get(l))
//    for (int i = 0; i < queue.size(); i++) {
//      Interval curInterval = queue.get(i);
//      if (val >= curInterval.start && val <= curInterval.end) {
//        // inside current interval, do nothing
//        break;
//      } else if (val == curInterval.end + 1) {
//        // adjacent to cur interval
//        if (i + 1 < queue.size() && val == queue.get(i + 1).start - 1) {
//          // adjacent to next interval
//          curInterval.end = queue.get(i + 1).end;
//          queue.remove(i + 1);
//
//        } else {
//          curInterval.end = val;
//        }
//        break;
//      } else if (val == curInterval.start - 1) {
//        curInterval.start = val;
//        break;
//      } else if (val < curInterval.start - 1) {
//        queue.add(i, new Interval(val, val));
//        break;
//      } else if (i == queue.size() - 1 && val > curInterval.end + 1) {
//        queue.add(new Interval(val, val));
//        break;
//      }
//    }
//  }
//
//  public List<Interval> getIntervals() {
//    return queue;
//  }
//}
