import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class RandomPickIndex {

  private Map<Integer, Main.IntegerPair> m;
  private Random r;

  public RandomPickIndex(int[] nums) {
    r = new Random();
    Arrays.sort(nums);
    m = new HashMap<>();
    int i = 0;
    int n = nums.length;
    while (i < n) {
      int cur = nums[i];
      int j = i + 1;
      while (j < n && nums[j] == cur) {
        j++;
      }
      m.put(cur, new Main.IntegerPair(i, j - 1));
      i = j;
    }
  }

  public int pick(int target) {
    Main.IntegerPair p = m.get(target);
    return r.nextInt(p.second - p.first + 1) + p.first;
  }
}
