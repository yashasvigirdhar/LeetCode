import java.util.*;

public class RandomizedSet {

  private Map<Integer, Integer> map;
  private int[] arr;

  /**
   * Initialize your data structure here.
   */
  public RandomizedSet() {
    map = new HashMap<>();
    arr = new int[10000];
  }

  /**
   * Inserts a value to the map. Returns true if the map did not already contain the specified element.
   */
  public boolean insert(int val) {
    if (map.containsKey(val)) {
      return false;
    }
    if (map.size() == arr.length/2) {
      int[] temp = new int[arr.length * 2];
      System.arraycopy(arr, 0, temp, 0, arr.length);
      arr = temp;
    }
    int newIdx = map.size();
    map.put(val, newIdx);
    arr[newIdx] = val;
    return true;
  }

  /**
   * Removes a value from the map. Returns true if the map contained the specified element.
   */
  public boolean remove(int val) {
    Integer idx = map.get(val);
    if (idx == null) {
      return false;
    }
    int lastIdx = map.size() - 1;
    int newValAtIdx = arr[lastIdx];
    arr[idx] = newValAtIdx;
    map.put(newValAtIdx, idx);
    map.remove(val);
    return true;
  }

  /**
   * Get a random element from the map.
   */
  public int getRandom() {
    return arr[new Random().nextInt(map.size())];
  }

}
