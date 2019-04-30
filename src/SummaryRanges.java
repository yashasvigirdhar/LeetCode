import java.util.ArrayList;
import java.util.List;

public class SummaryRanges {

  /**
   * Initialize your data structure here.
   */

  List<Interval> queue;

  public SummaryRanges() {
    queue = new ArrayList<>();
  }

  public void addNum(int val) {
    if (queue.size() == 0) {
      queue.add(new Interval(val, val));
      return;
    }
    for (int i = 0; i < queue.size(); i++) {
      Interval curInterval = queue.get(i);
      if (val >= curInterval.start && val <= curInterval.end) {
        // inside current interval, do nothing
        break;
      } else if (val == curInterval.end + 1) {
        // adjacent to cur interval
        if (i + 1 < queue.size() && val == queue.get(i + 1).start - 1) {
          // adjacent to next interval
          curInterval.end = queue.get(i + 1).end;
          queue.remove(i + 1);

        } else {
          curInterval.end = val;
        }
        break;
      } else if (val == curInterval.start - 1) {
        curInterval.start = val;
        break;
      } else if (val < curInterval.start - 1) {
        queue.add(i, new Interval(val, val));
        break;
      } else if (i == queue.size() - 1 && val > curInterval.end + 1) {
        queue.add(new Interval(val, val));
        break;
      }
    }
  }

  public List<Interval> getIntervals() {
    return queue;
  }
}
