import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

public class Codec2 {

  String DELIM = ",";
  String NULL = "#";

  // Encodes a tree to a single string.
  public String serialize(TreeNode root) {
    StringBuilder b = new StringBuilder();
    if (root == null) {
      return NULL;
    }
    serialize(root, b);
    return b.substring(0, b.length() - 1);
  }

  private void serialize(TreeNode root, StringBuilder b) {
    if (root == null) {
      b.append(NULL).append(DELIM);
      return;
    }
    b.append(root.val).append(DELIM);
    serialize(root.left, b);
    serialize(root.right, b);
  }

  // Decodes your encoded data to tree.
  public TreeNode deserialize(String data) {
    String[] split = data.split(DELIM);
    Deque<String> q = new ArrayDeque<>(Arrays.asList(split));
    return buildTree(q);
  }

  private TreeNode buildTree(Deque<String> q) {
    String s = q.pollFirst();
    if (s.equals(NULL)) {
      return null;
    } else {
      TreeNode root = new TreeNode(Integer.parseInt(s));
      root.left = buildTree(q);
      root.right = buildTree(q);
      return root;
    }
  }
}
