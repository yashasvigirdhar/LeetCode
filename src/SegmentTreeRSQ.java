public class SegmentTreeRSQ {

  private int[] tree;
  int n;

  public SegmentTreeRSQ(int[] nums) {
    n = nums.length;
    tree = new int[4 * n];
    buildTree(0, 0, n - 1, nums);
  }

  private void buildTree(int pos, int left, int right, int[] nums) {
    if (left == right) {
      tree[pos] = nums[left];
      return;
    }
    int mid = (left + right) / 2;
    buildTree(2 * pos + 1, left, mid, nums);
    buildTree(2 * pos + 2, mid + 1, right, nums);
    tree[pos] = tree[2 * pos + 1] + tree[2 * pos + 2];
  }

  int sumRange(int l, int r) {
    return sumRange(0, 0, n - 1, l, r);
  }

  private int sumRange(int pos, int l, int r, int i, int j) {

    // completely inside query range
    if (l >= i && r <= j) {
      return tree[pos];
    }
    int mid = (l + r) / 2;
    if (i > mid) {
      return sumRange(2 * pos + 2, mid + 1, r, i, j);
    } else if (j <= mid) {
      return sumRange(2 * pos + 1, l, mid, i, j);
    } else {
      return sumRange(2 * pos + 1, l, mid, i, j) + sumRange(2 * pos + 2, mid + 1, r, i, j);
    }
  }

  public void update(int i, int val) {
    update(0, 0, n - 1, i, val);
  }

  private void update(int pos, int l, int r, int idx, int val) {
    if (l == r) {
      tree[pos] = val;
      return;
    }
    int mid = (l + r) / 2;
    if (idx <= mid) {
      update(2 * pos + 1, l, mid, idx, val);
    } else {
      update(2 * pos + 2, mid + 1, r, idx, val);
    }
    tree[pos] = tree[2 * pos + 1] + tree[2 * pos + 2];
  }
}
