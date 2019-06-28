public class TreeWithSmallerElementsCount {

  TreeNodeWithSmaller root;

  public TreeWithSmallerElementsCount() {
  }

  int insert(int val) {
    int[] res = new int[1];
    root = insert(root, val, 0, res);
    return res[0];
  }

  private TreeNodeWithSmaller insert(TreeNodeWithSmaller root, int val, int count, int[] res) {
    if (root == null) {
      TreeNodeWithSmaller t = new TreeNodeWithSmaller(val);
      res[0] = count;
      return t;
    }

    if (val > root.val) {
      root.right = insert(root.right, val, count + root.numLeft + 1, res);
    } else {
      root.numLeft++;
      root.left = insert(root.left, val, count, res);
    }
    return root;
  }

  class TreeNodeWithSmaller {
    int val;
    int numLeft;
    TreeNodeWithSmaller left, right;

    public TreeNodeWithSmaller(int val) {
      this.val = val;
    }
  }
}


