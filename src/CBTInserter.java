import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

class CBTInserter {

  List<TreeNode> tree;

  public CBTInserter(TreeNode root) {
    tree = new ArrayList<>();
    tree.add(new TreeNode(-1));
    Deque<TreeNode> queue = new ArrayDeque<>();
    queue.addLast(root);
    while (!queue.isEmpty()){
      TreeNode polled = queue.pollFirst();
      tree.add(polled);
      if(polled.left != null){
        queue.addLast(polled.left);
      }
      if(polled.right != null){
        queue.addLast(polled.right);
      }
    }
  }

  public int insert(int v) {
    TreeNode t = new TreeNode(v);
    tree.add(t);
    int childIdx = tree.size()-1;
    int parentIdx = childIdx/2;
    TreeNode parent = tree.get(parentIdx);
    if(childIdx%2==0){
      parent.left = t;
    }else {
      parent.right = t;
    }
    return parent.val;
  }

  public TreeNode get_root() {
    return tree.size()>1 ? tree.get(1): null;
  }
}
