public class NumArraySegmentTree {

  private int[] tree;
  private int size;

  public NumArraySegmentTree(int[] nums) {

    tree = new int[nums.length * 4];
    size = nums.length;
    if (size > 0) {
      constructTree(nums, 0, 0, nums.length - 1);
    }
  }

  // order : O(2*n)
  private void constructTree(int[] nums, int pos, int l, int r) {
    if (l == r) {
      tree[pos] = nums[l];
      return;
    }
    int mid = l + (r - l) / 2;
    constructTree(nums, 2 * pos + 1, l, mid);
    constructTree(nums, 2 * pos + 2, mid + 1, r);
    tree[pos] = tree[2 * pos + 1] + tree[2 * pos + 2];
  }

  public void update(int i, int val) {
    updateUtil(0, 0, size - 1, i, val);
  }

  private void updateUtil(int pos, int left, int right, int i, int val) {
    if (left == right) {
      tree[pos] = val;
      return;
    }
    int mid = left + (right - left) / 2;
    if (i > mid) {
      updateUtil(2 * pos + 2, mid + 1, right, i, val);
    } else {
      updateUtil(2 * pos + 1, left, mid, i, val);
    }
    tree[pos] = tree[2 * pos + 1] + tree[2 * pos + 2];
  }

  public int sumRange(int i, int j) {
    return sumUtil(0, 0, size - 1, i, j);
  }

  private int sumUtil(int pos, int left, int right, int i, int j) {
    if (left == right) {
      return tree[pos];
    }
    if (j < left || i > right) {  // no overlap
      return 0;
    }
    if (left >= i && right <= j) {  // totalSize overlap
      return tree[pos];
    }
    int mid = left + (right - left) / 2;
    if (i > mid) {    // if range is only in right half
      return sumUtil(2 * pos + 2, mid + 1, right, i, j);
    } else if (j <= mid) {  // if range is only in left half
      return sumUtil(2 * pos + 1, left, mid, i, j);
    }
    return sumUtil(2 * pos + 1, left, mid, i, j) + sumUtil(2 * pos + 2, mid + 1, right, i, j);
  }

  public void printTree() {
    for (int element : tree) {
      System.out.print(element + " ");
    }
    System.out.println();
  }
}
