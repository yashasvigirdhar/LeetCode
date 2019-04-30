import java.util.Random;

public class ShuffleArray {

  int[] nums;

  public ShuffleArray(int[] nums) {
    this.nums = nums;
  }

  /**
   * Resets the array to its original configuration and return it.
   */
  public int[] reset() {
    return nums;
  }

  /**
   * Returns a random shuffling of the array.
   */
  public int[] shuffle() {
    int n = nums.length;
    int[] result = new int[n];
    System.arraycopy(nums, 0, result, 0, n);
    while (n > 1) {
      int r = new Random().nextInt(n);
      int t = result[r];
      result[r] = result[n - 1];
      result[n - 1] = t;
      n--;
    }
    return result;
  }
}
