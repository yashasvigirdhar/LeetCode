import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

class HitCounter {

  SortedMap<Integer, Integer> map;

  /**
   * Initialize your data structure here.
   */
  public HitCounter() {
    map = new TreeMap<>();
  }

  /**
   * Record a hit.
   *
   * @param timestamp - The current timestamp (in seconds granularity).
   */
  public void hit(int timestamp) {
    if (!map.containsKey(timestamp)) {
      map.put(timestamp, 0);
    }
    map.put(timestamp, map.get(timestamp) + 1);
  }

  /**
   * Return the number of hits in the past 5 minutes.
   *
   * @param timestamp - The current timestamp (in seconds granularity).
   */
  public int getHits(int timestamp) {
    int desiredTimestamp = timestamp - 300;

    int counter = 0;
    for (Map.Entry<Integer, Integer> it : map.tailMap(desiredTimestamp + 1).entrySet()) {
      if (it.getKey() > timestamp) {
        break;
      }
      counter += it.getValue();
    }
    return counter;
  }
}