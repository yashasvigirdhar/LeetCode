import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

public class Codec {

  private static final Character DELIMITER = '#';
  private static final Character NULL = '$';

  int idx = -1;

  // Encodes a tree to a single string.
  public String serialize(TreeNode root) {
    StringBuilder b = new StringBuilder();
    buildString(root, b);
    return b.toString();
  }

  private void buildString(TreeNode root, StringBuilder b) {
    if (root == null) {
      b.append(NULL).append(DELIMITER);
      return;
    }
    b.append(root.val).append(DELIMITER);
    buildString(root.left, b);
    buildString(root.right, b);
  }


  // Decodes your encoded data to tree.
  public TreeNode deserialize(String data) {
    List<String> strings = Arrays.asList(data.split(String.valueOf(DELIMITER)));
    Deque<String> queue = new LinkedList<>(strings);
    idx = 0;
    return buildTree(strings);
  }

  private TreeNode buildTree(List<String> nodes) {
    String val = nodes.get(idx++);
    if (val.equals(String.valueOf(NULL))) {
      return null;
    }
    TreeNode node = new TreeNode(Integer.parseInt(val));
    node.left = buildTree(nodes);
    node.right = buildTree(nodes);
    return node;
  }


}