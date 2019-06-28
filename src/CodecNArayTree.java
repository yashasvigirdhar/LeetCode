import java.util.*;

public class CodecNArayTree {

  private static final String START = "[";
  private static final String END = "]";
  private static final String DELIMITER = ",";

  // Encodes a tree to a single string.
  public String serialize(Node root) {
    StringBuilder b = new StringBuilder();
    if (root != null) {
      serialize(root, b);
      if (String.valueOf(b.charAt(b.length() - 1)).equals(DELIMITER)) {
        return b.substring(0, b.length() - 1);
      }
    }

    return b.toString();
  }

  private void serialize(Node root, StringBuilder b) {
    b.append(root.val).append(DELIMITER);
    if (root.children.size() > 0) {
      b.append(START).append(DELIMITER);
      for (Node child : root.children) {
        serialize(child, b);
      }
      b.append(END).append(DELIMITER);
    }
  }

  // Decodes your encoded data to tree.
  public Node deserialize(String data) {
    if (data.equals("")) {
      return null;
    }
    String[] split = data.split(DELIMITER);
    Deque<String> q = new ArrayDeque<>(Arrays.asList(split));
    Node root = new Node(Integer.parseInt(q.pollFirst()), new ArrayList<>());
    Deque<Node> parents = new ArrayDeque<>();
    Node lastPolled = root;
    while (!q.isEmpty()) {
      String polled = q.poll();
      if (polled.equals(START)) {
        parents.addFirst(lastPolled);
      } else if (polled.equals(END)) {
        parents.pollFirst();
      } else {
        Node n = new Node(Integer.parseInt(polled), new ArrayList<>());
        lastPolled = n;
        parents.peek().children.add(n);
      }
    }
    return root;
  }
}

class Node {
  public int val;
  public List<Node> children;

  public Node() {
  }

  public Node(int _val, List<Node> _children) {
    val = _val;
    children = _children;
  }
}

