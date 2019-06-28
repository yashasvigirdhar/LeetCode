class MyCalenderThree {

  private SegmentTree tree;

  public MyCalenderThree() {
    tree = new SegmentTree(1, 100000000);
  }

  public int book(int start, int end) {
    tree.add(start, end, 1);
    return tree.getMax(tree.root);
  }

  class SegmentTree {

    SegmentTreeroot root;

    public SegmentTree(int l, int r) {
      root = new SegmentTreeroot(l, r);
    }

    public void add(int l, int r, int val) {
      add(root, l, r, val);
    }

    private void add(SegmentTreeroot root, int start, int end, int val) {

      //no overlap
      if (root == null || start >= root.end || end < root.start) return;

      // completely inside the query range
      if (root.start >= start && root.end <= end) {
        root.count += val;
        root.maxCount += val;
        return;
      }

      int mid = root.start + (root.end - root.start) / 2;
      if (overlap(root.start, mid, start, end)) {
        if (root.left == null) root.left = new SegmentTreeroot(root.start, mid);
        add(root.left, start, end, val);
      }

      if (overlap(mid, root.end, start, end)) {
        if (root.right == null) root.right = new SegmentTreeroot(mid, root.end);
        add(root.right, start, end, val);
      }

      root.maxCount = Math.max(getMax(root.left), getMax(root.right)) + root.count;
    }

    private int getMax(SegmentTreeroot root) {
      if (root == null) return 0;
      return root.maxCount;
    }
  }

  private boolean overlap(int s, int e, int l, int r) {
    if (r <= s || l >= e) return false;
    return true;
  }

  class SegmentTreeroot {
    int start, end;
    int count = 0;
    int maxCount = 0;

    SegmentTreeroot left, right;

    public SegmentTreeroot(int start, int end) {
      this.start = start;
      this.end = end;
    }
  }
}




