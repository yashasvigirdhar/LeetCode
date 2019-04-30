class NumArray {

  private int[] sums;
  private int[] nums;

  public NumArray(int[] nums) {
    if (nums.length == 0) {
      return;
    }
    this.nums = nums;
    sums[0] = this.nums[0];
    for (int i = 1; i < this.nums.length; i++) {
      sums[i] = sums[i - 1] + this.nums[i];
    }
  }

  public int sumRange(int i, int j) {
    if (i == 0) {
      return sums[j];
    }
    return sums[j] - sums[i - 1];
  }

  public void update(int i, int val) {
    int diff = val - nums[i];
    nums[i] = val;
    for (int j = i; j < sums.length; j++) {
      sums[i] += diff;
    }
  }
}