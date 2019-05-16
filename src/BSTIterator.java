import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

class BSTIterator {

  private Stack<TreeNode> stack;
  private Set<TreeNode> set;

  public BSTIterator(TreeNode root) {
    stack = new Stack<>();
    set = new HashSet<>();
    if (root != null) {
      fillStack(root);
    }
  }

  private void fillStack(TreeNode root) {
    stack.push(root);
    while (root.left != null) {
      root = root.left;
      stack.push(root);
    }
  }

  /**
   * @return the next smallest number
   */
  public int next() {
    TreeNode popped = stack.pop();
    if (popped.right != null) {
      fillStack(popped.right);
    }
    return popped.val;
  }

  /**
   * @return whether we have a next smallest number
   */
  public boolean hasNext() {
    return !stack.isEmpty();
  }
}
