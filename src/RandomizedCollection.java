import java.util.*;

public class RandomizedCollection {

  private Map<Integer, List<Integer>> map;
  private int[] arr;
  int totalCapacity = 0;

  /**
   * Initialize your data structure here.
   */
  public RandomizedCollection() {
    map = new HashMap<>();
    arr = new int[10000];
  }

  /**
   * Inserts a value to the map. Returns true if the map did not already contain the specified element.
   */
  public boolean insert(int val) {
    boolean res = true;
    if (map.containsKey(val)) {
      res = false;
    } else {
      map.put(val, new ArrayList<>());
    }
    if (map.size() == arr.length / 2) {
      int[] temp = new int[arr.length * 2];
      System.arraycopy(arr, 0, temp, 0, arr.length);
      arr = temp;
    }

    int newIdx = totalCapacity++;
    map.get(val).add(newIdx);
    arr[newIdx] = val;

    return res;
  }

  /**
   * Removes a value from the map. Returns true if the map contained the specified element.
   */
  public boolean remove(int val) {
    List<Integer> indices = map.get(val);
    if (indices == null) {
      return false;
    }
    int lastIdx = totalCapacity - 1;
    int newValAtIdx = arr[lastIdx];

    if (val == newValAtIdx) {
      // just remove the last idx;
      indices.remove((Integer) lastIdx);
      if (indices.size() == 0) {
        map.remove(val);
      }
      totalCapacity--;
      return true;
    }

    Integer idxToShift = indices.remove(0);

    List<Integer> newIndices = map.get(newValAtIdx);
    newIndices.remove((Integer) lastIdx);
    arr[idxToShift] = newValAtIdx;
    newIndices.add(idxToShift);

    if (indices.size() == 0) {
      //remove entry if last index
      map.remove(val);
    }
    totalCapacity--;
    return true;
  }

  /**
   * Get a random element from the map.
   */
  public int getRandom() {
    return arr[new Random().nextInt(totalCapacity)];
  }
}
