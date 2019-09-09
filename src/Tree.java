import java.util.*;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

public class Tree {

  public List<List<Integer>> findLeaves(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    while (root != null) {
      List<Integer> cur = new ArrayList<>();
      root = findLeaveUtil(root, cur);
      res.add(cur);
    }
    return res;
  }

  private TreeNode findLeaveUtil(TreeNode root, List<Integer> cur) {
    if (root == null) {
      return null;
    }
    if (root.left == null && root.right == null) {
      cur.add(root.val);
      return null;
    }
    root.left = findLeaveUtil(root.left, cur);
    root.right = findLeaveUtil(root.right, cur);
    return root;
  }

  public int maxPathSumMock(TreeNode root) {
    int[] res = new int[]{Integer.MIN_VALUE};
    maxPathSumUtil(root, res);
    return res[0];
  }

  private int maxPathSumUtil(TreeNode root, int[] res) {
    if (root == null) {
      return 0;
    }
    int l = Math.max(0, maxPathSumUtil(root.left, res));
    int r = Math.max(0, maxPathSumUtil(root.right, res));
    res[0] = Math.max(res[0], l + r + root.val);
    return Math.max(l, r) + root.val;
  }

  public TreeNode lcaDeepestLeaves2(TreeNode root) {
    int[] maxDepth = new int[]{-1};
    calcDepth(root, 0, maxDepth);

    return findLcaV2(root, 0, maxDepth[0]);
  }

  private TreeNode findLcaV2(TreeNode root, int depth, int maxDepth) {
    if (root == null) {
      return null;
    }
    if (depth == maxDepth) {
      return root;
    }
    TreeNode left = findLcaV2(root.left, depth + 1, maxDepth);
    TreeNode right = findLcaV2(root.right, depth + 1, maxDepth);
    if (left != null && right != null) {
      return root;
    }
    if (left != null) {
      return left;
    }
    return right;
  }

  private void calcDepth(TreeNode root, int depth, int[] maxDepth) {
    if (root == null) {
      return;
    }
    if (root.left == null && root.right == null) {
      if (depth > maxDepth[0]) {
        maxDepth[0] = depth;
      }
      return;
    }
    calcDepth(root.left, depth + 1, maxDepth);
    calcDepth(root.right, depth + 1, maxDepth);
  }

  public TreeNode lcaDeepestLeaves(TreeNode root) {
    TreeNode[] parents = new TreeNode[1001];
    populateParentsAgain(root, parents, null);

    Set<TreeNode> leaves = new HashSet<>();
    int[] maxDepth = new int[]{-1};
    calcDepth(root, 0, maxDepth, leaves);

    Iterator<TreeNode> iterator = leaves.iterator();
    TreeNode lca = iterator.next();
    while (iterator.hasNext()) {
      lca = findLCA(lca, iterator.next(), parents);
    }
    return lca;
  }

  private TreeNode findLCA(TreeNode lca, TreeNode t2, TreeNode[] parents) {
    int[] firstPath = new int[1001];
    while (lca != null) {
      firstPath[lca.val] = 1;
      lca = parents[lca.val];
    }

    while (t2 != null) {
      if (firstPath[t2.val] == 1) {
        return t2;
      }
      t2 = parents[t2.val];
    }
    return null;
  }

  private void calcDepth(TreeNode root, int depth, int[] maxDepth, Set<TreeNode> leaves) {
    if (root == null) {
      return;
    }
    if (root.left == null && root.right == null) {
      if (depth > maxDepth[0]) {
        maxDepth[0] = depth;
        leaves.clear();
        leaves.add(root);
      } else if (depth == maxDepth[0]) {
        leaves.add(root);
      }
      return;
    }
    calcDepth(root.left, depth + 1, maxDepth, leaves);
    calcDepth(root.right, depth + 1, maxDepth, leaves);
  }

  private void populateParentsAgain(TreeNode root, TreeNode[] parents, TreeNode parent) {
    if (root == null) {
      return;
    }
    parents[root.val] = parent;
    populateParentsAgain(root.left, parents, root);
    populateParentsAgain(root.right, parents, root);
  }

  public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
    int[] deleted = new int[1001];
    for (int d : to_delete) {
      deleted[d] = 1;
    }
    ArrayList<TreeNode> res = new ArrayList<>();
    populateParentsMan(root, null, deleted, res);
    return res;
  }

  private void populateParentsMan(TreeNode root, TreeNode parent, int[] deleted, List<TreeNode> res) {
    if (root == null) {
      return;
    }
    if ((parent == null || deleted[parent.val] == 1) && deleted[root.val] == 0) {
      // add to res
      res.add(root);
    }

    if (root.left != null) {
      populateParentsMan(root.left, root, deleted, res);
      if (deleted[root.left.val] == 1) {
        root.left = null;
      }
    }

    if (root.right != null) {
      populateParentsMan(root.right, root, deleted, res);
      if (deleted[root.right.val] == 1) {
        root.right = null;
      }
    }

  }

  public String smallestFromLeaf(TreeNode root) {
    String[] res = new String[]{""};
    if (root == null) return res[0];
    smallestFromLeaf(root, new StringBuilder(), res);
    return res[0];
  }

  private void smallestFromLeaf(TreeNode root, StringBuilder cur, String[] res) {
    if (root.left == null && root.right == null) {
      cur.insert(0, (char) (root.val + 'a'));
      String s = cur.toString();
      if (res[0].length() == 0 || s.compareTo(res[0]) < 0) {
        res[0] = s;
      }
      cur.deleteCharAt(0);
      return;
    }
    cur.insert(0, (char) (root.val + 'a'));
    if (root.left != null) {
      smallestFromLeaf(root.left, cur, res);
    }
    if (root.right != null) {
      smallestFromLeaf(root.right, cur, res);
    }
    cur.deleteCharAt(0);
  }

  public int distributeCoinsPractice(TreeNode root) {
    int[] res = new int[0];
    distributeCoinsUtil(root, res);
    return res[0];
  }

  private int distributeCoinsUtil(TreeNode root, int[] res) {
    if (root == null) {
      return 0;
    }
    int left = distributeCoinsUtil(root.left, res);
    int right = distributeCoinsUtil(root.left, res);
    res[0] += Math.abs(left);
    res[0] += Math.abs(right);
    int totalCoins = root.val + (left > 0 ? left : 0) + (right > 0 ? right : 0);
    int requiredCoins = 1 + (left < 0 ? Math.abs(left) : 0) + (right < 0 ? Math.abs(right) : 0);
    return totalCoins - requiredCoins;
  }

  public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null) {
      return root;
    }
    int smallVal = Math.min(p.val, q.val);
    int largeVal = Math.max(p.val, q.val);

    if (root.val < smallVal) {
      return lowestCommonAncestorBST(root.right, p, q);
    }
    if (root.val > largeVal) {
      return lowestCommonAncestorBST(root.left, p, q);
    }

    if (root.val >= smallVal && root.val <= largeVal) {
      return root;
    }
    return root;
  }

  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    Map<TreeNode, TreeNode> parents = new HashMap<>();
    populateParents(root, parents, null);
    Set<TreeNode> parent1 = new HashSet<>();
    TreeNode tempParent = p;
    while (tempParent != null) {
      parent1.add(tempParent);
      tempParent = parents.get(tempParent);
    }
    while (q != null) {
      if (parent1.contains(q)) {
        return q;
      }
      q = parents.get(q);
    }
    return null;
  }

  private void populateParents(TreeNode t, Map<TreeNode, TreeNode> parents, TreeNode parent) {
    if (t == null) {
      return;
    }
    parents.put(t, parent);
    populateParents(t.left, parents, t);
    populateParents(t.right, parents, t);
  }

  public int findBottomLeftValue(TreeNode root) {
    int[] res = new int[]{-1, -1, -1};
    findBottomLeftValueUtil(root, 0, 0, res);
    return res[2];
  }

  private void findBottomLeftValueUtil(TreeNode root, int width, int depth, int[] res) {
    if (root == null) {
      return;
    }
    if (depth == res[0]) {
      if (width < res[1]) {
        res[0] = depth;
        res[1] = width;
        res[2] = root.val;
      }
    }
    if (depth > res[0]) {
      res[0] = depth;
      res[1] = width;
      res[2] = root.val;
    }

    findBottomLeftValueUtil(root.left, width - 1, depth + 1, res);
    findBottomLeftValueUtil(root.right, width + 1, depth + 1, res);
  }

  public TreeNode sufficientSubset(TreeNode root, int limit) {
    if (sufficientSubsetUtil(root, limit, 0)) {
      return null;
    }
    return root;
  }

  private boolean sufficientSubsetUtil(TreeNode root, int limit, int curSum) {
    if (root == null) {
      return true;
    }
    if (root.left == null && root.right == null) {
      return curSum + root.val < limit;
    }
    boolean l = sufficientSubsetUtil(root.left, limit, curSum + root.val);

    if (l) {
      root.left = null;
    }

    boolean r = sufficientSubsetUtil(root.right, limit, curSum + root.val);

    if (r) {
      root.right = null;
    }
    return l && r;
  }

  // for each node, caches the result for each edge. Each index in the array corresponds to one node.
  // each index contains a map where each entry (neighbour) corresponds to it's computed result, which is a
  // 2D array ( this 2D array is explained below).
  Map<Integer, int[]>[] dCache;

  public int[] sumOfDistancesInTree(int n, int[][] edges) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }

    for (int[] edge : edges) {
      graph.get(edge[0]).add(edge[1]);
      graph.get(edge[1]).add(edge[0]);
    }
    dCache = new HashMap[n];
    int[] distances = new int[n];
    Arrays.fill(distances, 0);
    for (int i = 0; i < n; i++) {
      distances[i] = sumOfDistanceInTree(i, -1, graph)[0];
    }
    return distances;
  }

  /**
   * @param idx
   * @param parent
   * @param graph
   * @return a 2D array containing the result of idx, and count of nodes of subtree rooted at idx.
   */
  private int[] sumOfDistanceInTree(int idx, int parent, List<List<Integer>> graph) {
    int res = 0;
    int numOfNodes = 1;
    for (int neighbour : graph.get(idx)) {
      if (neighbour == parent) {
        continue;
      }
      int[] t;
      if (dCache[idx] == null) {
        dCache[idx] = new HashMap<>();
      }
      // check if we have already traversed the edge from idx to neighbour
      if (dCache[idx].containsKey(neighbour)) {
        t = dCache[idx].get(neighbour);
      } else {
        t = sumOfDistanceInTree(neighbour, idx, graph);
        dCache[idx].put(neighbour, t);
      }
      numOfNodes += t[1]; // add the no of nodes
      res += (t[0] + t[1]); // add the result of that subtree. for every node, you add 1 to the result of neighbour.
    }

    return new int[]{res, numOfNodes};
  }

  int minWidth, maxWidth;

  public List<List<Integer>> verticalTraversal(TreeNode root) {
    if (root == null) {
      return new ArrayList<>();
    }
    Map<Integer, List<Pair<Integer, Integer>>> map = new HashMap<>();
    minWidth = 0;
    maxWidth = 0;
    verticalTraversalUtil(root, 0, 0, map);
    List<List<Integer>> res = new ArrayList<>();
    for (int i = minWidth; i <= maxWidth; i++) {
      List<Pair<Integer, Integer>> list = map.get(i);
      list.sort(new Comparator<Pair<Integer, Integer>>() {
        @Override
        public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
          if (o1.second.equals(o2.second)) {
            return o1.first - o2.first;
          }
          return o1.second - o2.second;
        }
      });
      List<Integer> l = new ArrayList<>();
      for (Pair<Integer, Integer> p : list) {
        l.add(p.first);
      }
      res.add(l);
    }
    return res;
  }

  private void verticalTraversalUtil(TreeNode root, int width, int depth, Map<Integer, List<Pair<Integer, Integer>>> map) {
    if (root == null) {
      return;
    }
    minWidth = Math.min(minWidth, width);
    maxWidth = Math.max(maxWidth, width);
    map.computeIfAbsent(width, k -> new ArrayList<>());
    map.get(width).add(new Pair<>(root.val, depth));
    verticalTraversalUtil(root.left, width - 1, depth + 1, map);
    verticalTraversalUtil(root.right, width + 1, depth + 1, map);
  }

  public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
    Set<TreeNode> res = new HashSet<>();
    serializeSubtrees(root, new HashMap<>(), res);
    return new ArrayList<>(res);
  }

  private String serializeSubtrees(TreeNode root, Map<String, TreeNode> map, Set<TreeNode> res) {
    if (root == null) {
      return "#";
    }
    String cur = root.val + serializeSubtrees(root.left, map, res) + serializeSubtrees(root.right, map, res);
    if (map.containsKey(cur)) {
      res.add(map.get(cur));
    } else {
      map.put(cur, root);
    }
    return cur;
  }

  public NodeParent inorderSuccessor(NodeParent p) {
    NodeParent ans = null;
    if (p.right != null) {
      ans = p.right;
      while (ans.left != null) {
        ans = ans.left;
      }
      return ans;
    }
    while (p.parent != null) {
      if (p.parent.left == p) {
        ans = p.parent;
        break;
      }
      p = p.parent;
    }
    return ans;
  }

  class NodeParent {
    public int val;
    public NodeParent left;
    public NodeParent right;
    public NodeParent parent;
  }

  ;

  public TreeNode inorderSuccessorII(TreeNode root, TreeNode p) {
    TreeNode succ = null;
    while (root != null) {
      if (p.val < root.val) {
        succ = root;
        root = root.left;
      } else {
        root = root.right;
      }
    }
    return succ;
  }

  public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
    if (root == null) {
      return null;
    }
    if (p.right != null) {
      TreeNode ans = p.right;
      while (ans.left != null) {
        ans = ans.left;
      }
      return ans;
    }
    Stack<TreeNode> st = new Stack<>();
    boolean pCrossed = false;
    fillStack(st, root, p);
    while (!st.empty()) {
      TreeNode popped = st.pop();
      if (pCrossed) {
        return popped;
      }
      if (popped == p) {
        pCrossed = true;
      }
      if (popped.right != null) {
        fillStack(st, popped.right, p);
      }
    }
    return null;
  }

  private void fillStack(Stack<TreeNode> stack, TreeNode root, TreeNode p) {
    stack.push(root);
    while (root != p && root.left != null) {
      root = root.left;
      stack.push(root);
    }
  }

  public int maxDepth(Node root) {
    if (root == null) {
      return 0;
    }
    int maxFromChildren = 0;
    for (int i = 0; i < root.children.size(); i++) {
      maxFromChildren = Math.max(maxFromChildren, maxDepth(root.children.get(i)));
    }
    return maxFromChildren + 1;
  }

  public List<List<Integer>> levelOrder(Node root) {
    List<List<Integer>> res = new ArrayList<>();
    if (root == null) {
      return res;
    }

    List<Integer> arr = new ArrayList<>();
    arr.add(root.val);
    res.add(arr);

    Queue<Node> queue = new LinkedList<>();
    queue.add(root);
    while (queue.size() > 0) {
      List<Node> temp = new ArrayList<>();
      List<Integer> vals = new ArrayList<>();
      while (!queue.isEmpty()) {
        Node polled = queue.poll();
        for (int i = 0; i < polled.children.size(); i++) {
          Node child = polled.children.get(i);
          if (child != null) {
            temp.add(child);
            vals.add(child.val);
          }
        }
      }
      if (temp.size() > 0) {
        res.add(vals);
        queue.addAll(temp);
      }
    }
    return res;
  }

  public List<Integer> postorder(Node root) {
    List<Integer> res = new ArrayList<>();
    Stack<Node> st = new Stack<>();
    st.push(root);
    while (!st.empty()) {
      Node popped = st.pop();
      res.add(popped.val);
      int i = 0;
      while (i < popped.children.size()) {
        st.push(popped.children.get(i));
        i++;
      }
    }
    Collections.reverse(res);
    return res;
  }

  public List<Integer> preorderIterative(Node root) {
    List<Integer> res = new ArrayList<>();
    Stack<Node> st = new Stack<>();
    st.push(root);
    while (!st.empty()) {
      Node popped = st.pop();
      res.add(popped.val);
      int i = popped.children.size() - 1;
      while (i >= 0) {
        st.push(popped.children.get(i));
        i--;
      }
    }
    return res;
  }

  public List<Integer> preorder(Node root) {
    List<Integer> res = new ArrayList<>();
    preOrderUtil(root, res);
    return res;
  }

  private void preOrderUtil(Node root, List<Integer> res) {
    if (root == null) {
      return;
    }
    res.add(root.val);
    for (int i = 0; i < root.children.size(); i++) {
      preOrderUtil(root.children.get(i), res);
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

  ;

  public List<Double> averageOfLevels(TreeNode root) {
    List<Double> res = new LinkedList<>();
    if (root == null) {
      return res;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while (queue.size() > 0) {
      List<TreeNode> temp = new ArrayList<>();
      long curSum = 0, count = 0;
      while (!queue.isEmpty()) {
        TreeNode polled = queue.poll();
        count++;
        curSum += polled.val;
        if (polled.left != null) {
          temp.add(polled.left);
        }
        if (polled.right != null) {
          temp.add(polled.right);
        }
      }
      res.add((double) curSum / (double) count);
      queue.addAll(temp);
    }
    return res;
  }

  public TreeNode invertTreePractice(TreeNode root) {
    if (root == null) {
      return root;
    }
    root.left = invertTreePractice(root.left);
    root.right = invertTreePractice(root.right);
    TreeNode t = root.left;
    root.left = root.right;
    root.right = t;
    return root;
  }

  int csum = 0;

  public TreeNode bstToGst(TreeNode root) {
    if (root == null) {
      return root;
    }
    bstToGst(root.right);
    csum += root.val;
    root.val = csum;
    bstToGst(root.left);
    return root;
  }

  public int widthOfBinaryTreeOptimized(TreeNode root) {
    if (root == null) {
      return 0;
    }
    Queue<Pair<TreeNode, Integer>> queue = new LinkedList<>();
    queue.add(new Pair<>(root, 0));
    int maxLen = 1;
    while (!queue.isEmpty()) {
      List<Pair<TreeNode, Integer>> temp = new ArrayList<>();

      while (!queue.isEmpty()) {
        Pair<TreeNode, Integer> poll = queue.poll();
        TreeNode parent = poll.first;
        int parentIdx = poll.second;
        if (parent.left != null) {
          temp.add(new Pair<>(parent.left, 2 * parentIdx));
        }
        if (parent.right != null) {
          temp.add(new Pair<>(parent.right, 2 * parentIdx + 1));
        }
      }

      if (temp.isEmpty()) {
        break;
      }
      maxLen = Math.max(maxLen, temp.get(temp.size() - 1).second - temp.get(0).second + 1);
      queue.addAll(temp);
    }
    return maxLen;
  }

  public int widthOfBinaryTree(TreeNode root) {
    if (root == null) {
      return 0;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    int maxLen = 1;
    while (!queue.isEmpty()) {
      List<TreeNode> temp = new ArrayList<>();
      while (!queue.isEmpty()) {
        TreeNode polled = queue.poll();
        if (polled != null) {
          temp.add(polled.left);
          temp.add(polled.right);
        } else {
          temp.add(null);
          temp.add(null);
        }
      }

      int firstIdx = -1;
      int lastIdx = -1;
      for (int i = 0; i < temp.size(); i++) {
        if (temp.get(i) != null) {
          if (firstIdx == -1) {
            firstIdx = i;
            lastIdx = i;
          } else {
            lastIdx = i;
          }
        }
      }
      if (firstIdx == -1) {
        break;
      }
      int curLen = lastIdx - firstIdx + 1;
      maxLen = Math.max(maxLen, curLen);
      for (int i = firstIdx; i <= lastIdx; i++) {
        queue.add(temp.get(i));
      }
    }
    return maxLen;
  }

  public List<Integer> largestValues(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if (root == null) {
      return res;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while (!queue.isEmpty()) {
      List<TreeNode> temp = new ArrayList<>();
      int maxTemp = Integer.MIN_VALUE;
      while (!queue.isEmpty()) {
        TreeNode polled = queue.poll();
        maxTemp = Math.max(maxTemp, polled.val);
        if (polled.left != null) {
          temp.add(polled.left);
        }
        if (polled.right != null) {
          temp.add(polled.right);
        }
      }
      queue.addAll(temp);
      res.add(maxTemp);
    }
    return res;
  }

  public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
    Map<TreeNode, TreeNode> parentMap = new HashMap<>();
    populateParent(root, null, parentMap);
    Map<TreeNode, Integer> distances = new HashMap<>();
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(target);
    distances.put(target, 0);
    List<Integer> res = new ArrayList<>();
    while (!queue.isEmpty()) {
      TreeNode polled = queue.poll();
      Integer curDistance = distances.get(polled);
      if (curDistance == K) {
        res.add(polled.val);
        continue;
      }
      if (polled.left != null && curDistance + 1 <= K && curDistance + 1 < distances.getOrDefault(polled.left, Integer.MAX_VALUE)) {
        distances.put(polled.left, curDistance + 1);
        queue.add(polled.left);
      }
      if (polled.right != null && curDistance + 1 <= K && curDistance + 1 < distances.getOrDefault(polled.right, Integer.MAX_VALUE)) {
        distances.put(polled.right, curDistance + 1);
        queue.add(polled.right);
      }
      TreeNode curParent = parentMap.get(polled);
      if (curParent != null && curDistance + 1 <= K && curDistance + 1 < distances.getOrDefault(curParent, Integer.MAX_VALUE)) {
        distances.put(curParent, curDistance + 1);
        queue.add(curParent);
      }
    }
    return res;
  }

  private void populateParent(TreeNode cur, TreeNode par, Map<TreeNode, TreeNode> parentMap) {
    if (cur == null) {
      return;
    }
    parentMap.put(cur, par);
    populateParent(cur.left, cur, parentMap);
    populateParent(cur.right, cur, parentMap);
  }

  public TreeNode insertIntoBST(TreeNode root, int val) {
    if (root == null) {
      root = new TreeNode(val);
      return root;
    }
    if (val > root.val) {
      root.right = insertIntoBST(root.right, val);
    } else {
      root.left = insertIntoBST(root.left, val);
    }
    return root;
  }

  public List<Integer> rightSideView(TreeNode root) {
    ArrayList<Integer> ans = new ArrayList<>();
    rightSideView(root, 0, new HashSet<>(), ans);
    return ans;
  }

  private void rightSideView(TreeNode root, int depth, HashSet<Integer> depths, List<Integer> ans) {
    if (root == null) {
      return;
    }
    if (!depths.contains(depth)) {
      depths.add(depth);
      ans.add(root.val);
    }
    rightSideView(root.right, depth + 1, depths, ans);
    rightSideView(root.left, depth + 1, depths, ans);
  }

  public List<Integer> boundaryOfBinaryTree(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if (root == null) {
      return res;
    }
    res.add(root.val);
    TreeNode bottomLeftLeaf = null;
    if (root.left != null) {
      TreeNode node = root.left;
      while (node != null) {
        res.add(node.val);
        bottomLeftLeaf = node;
        if (node.left != null) {
          node = node.left;
        } else {
          node = node.right;
        }
      }
    }


    TreeNode bottomRightLeaf = null;
    List<TreeNode> right = new ArrayList<>();
    if (root.right != null) {
      TreeNode node = root.right;
      while (node != null) {
        right.add(node);
        bottomRightLeaf = node;
        if (node.right != null) {
          node = node.right;
        } else {
          node = node.left;
        }
      }
    }


    Queue<TreeNode> pq = new LinkedList<>();
    fillLeaves(root, pq);
    while (!pq.isEmpty()) {
      TreeNode poll = pq.poll();
      if (poll != root && poll != bottomLeftLeaf && poll != bottomRightLeaf) {
        res.add(poll.val);
      }
    }

    for (int i = right.size() - 1; i >= 0; i--) {
      res.add(right.get(i).val);
    }
    return res;
  }

  private void fillLeaves(TreeNode root, Queue<TreeNode> queue) {
    if (root == null) {
      return;
    }
    if (root.left == null && root.right == null) {
      queue.add(root);
      return;
    }
    fillLeaves(root.left, queue);
    fillLeaves(root.right, queue);

  }

  public boolean isSymmetricII(TreeNode root) {
    if (root == null) {
      return true;
    }
    return isSymmetricUtilII(root.left, root.right);
  }

  public boolean isSymmetricUtilII(TreeNode root1, TreeNode root2) {
    if (root1 == null && root2 == null) {
      return true;
    } else if (root1 != null && root2 != null) {
      return root1.val == root2.val && isSymmetricUtilII(root1.left, root2.right) && isSymmetricUtilII(root1.right, root2.left);
    } else {
      return false;
    }
  }


  private static void test() {
    TreeNode root = new TreeNode(3);
    root.left = new TreeNode(5);
    root.right = new TreeNode(1);
    root.right.left = new TreeNode(0);
    root.right.right = new TreeNode(8);
    root.left.left = new TreeNode(6);
    root.left.right = new TreeNode(2);
    root.left.right.left = new TreeNode(7);
    root.left.right.right = new TreeNode(4);
    System.out.print(lca(root, 4, 0));
  }

  public List<Integer> closestKValues(TreeNode root, double target, int k) {
    LinkedList<Integer> queue = new LinkedList<>();
    closestKValues(root, target, k, queue);
    List<Integer> ans = new ArrayList<>();
    while (!queue.isEmpty()) {
      ans.add(queue.poll());
    }
    return ans;
  }

  private void closestKValues(TreeNode root, double target, int k, LinkedList<Integer> queue) {
    if (root == null) {
      return;
    }
    closestKValues(root.left, target, k, queue);
    double curDiff = Math.abs(target - root.val);
    if (queue.size() < k) {
      queue.add(root.val);
    } else {
      if (curDiff < Math.abs(queue.peek() - target)) {
        queue.removeFirst();
        queue.add(root.val);
      } else {
        return;
      }
    }
    closestKValues(root.right, target, k, queue);
  }


  public int closestValueII(TreeNode root, double target) {
    int[] ans = new int[]{-1};
    closestValueUtil(root, target, ans, Double.MAX_VALUE);
    return ans[0];
  }

  private void closestValueUtil(TreeNode root, double target, int[] ans, double curMinDiff) {
    if (root == null) {
      return;
    }
    double curDiff = Math.abs(target - root.val);
    if (curDiff < curMinDiff) {
      curMinDiff = curDiff;
      ans[0] = root.val;
    }
    if (target > root.val) {
      closestValueUtil(root.right, target, ans, curMinDiff);
    } else if (target < root.val) {
      closestValueUtil(root.left, target, ans, curMinDiff);
    }
  }


  public int closestValue(TreeNode root, double target) {
    if (root == null) {
      return -1;
    }
    double curDiff = Math.abs(target - root.val);
    double childDiff = Double.MAX_VALUE;
    int childVal = 0;
    if (target > root.val) {
      childVal = closestValue(root.right, target);
      if (childVal != -1) {
        childDiff = Math.abs(target - childVal);
      }
    } else if (target < root.val) {
      childVal = closestValue(root.left, target);
      if (childVal != -1) {
        childDiff = Math.abs(target - childVal);
      }
    }
    if (curDiff <= childDiff) {
      return root.val;
    } else {
      return childVal;
    }
  }

  public int countNodesII(TreeNode root) {
    int h = height(root);
    int total = 0;
    while (root != null) {
      if (height(root.right) == h - 1) {
        total += (1 << h);
        root = root.right;
      } else {
        total += (1 << (h - 1));
        root = root.left;
      }
      h--;
    }
    return total;
  }

  private int height(TreeNode root) {
    if (root == null) return 0;
    return 1 + height(root.left);
  }

  int totalCount = 0;

  public int countNodes(TreeNode root) {
    if (root == null) {
      return 0;
    }
    totalCount++;
    countNodes(root.left);
    countNodes(root.right);
    return totalCount;
  }


  public TreeNode recoverFromPreorder(String s) {
    return recoverFromPreorder(s, 1);
  }


  public TreeNode recoverFromPreorder(String s, int curDepth) {
    if (s.length() == 0) {
      return null;
    }
    int idx = 0;
    while (idx < s.length() && s.charAt(idx) != '-') {
      idx++;
    }

    int rootVal = Integer.parseInt(s.substring(0, idx));
    TreeNode root = new TreeNode(rootVal);
    if (idx == s.length()) {
      return root;
    }

    int leftChildIdx = -1, rightChildIdx = -1;
    int count = 0;
    for (int i = idx; i < s.length(); i++) {
      if (s.charAt(i) == '-') {
        count++;
      } else {
        if (count == curDepth) {
          if (leftChildIdx == -1) {
            leftChildIdx = i;
          } else {
            rightChildIdx = i;
          }
        }
        count = 0;
      }
    }
    if (leftChildIdx != -1) {
      String substring;
      if (rightChildIdx == -1) {
        substring = s.substring(leftChildIdx);
      } else {
        substring = s.substring(leftChildIdx, rightChildIdx - curDepth);
      }
      root.left = recoverFromPreorder(substring, curDepth + 1);
    }
    if (rightChildIdx != -1) {
      root.right = recoverFromPreorder(s.substring(rightChildIdx), curDepth + 1);
    }
    return root;
  }

  public int maxAncestorDiffII(TreeNode root) {
    return maxAncestorDiffII(root, root.val, root.val);
  }

  public int maxAncestorDiffII(TreeNode root, int mx, int mn) {
    if (root == null) {
      return 0;
    }
    int res = mx - root.val;
    res = Math.max(res, root.val - mn);
    mx = Math.max(mx, root.val);
    mn = Math.min(mn, root.val);
    res = Math.max(res, maxAncestorDiffII(root.left, mx, mn));
    res = Math.max(res, maxAncestorDiffII(root.left, mx, mn));
    return res;
  }


  private int maxDiff = 0;

  public int maxAncestorDiff(TreeNode root) {
    maxAncestorDiff(root, new ArrayList<>());
    return maxDiff;
  }

  private void maxAncestorDiff(TreeNode root, List<Integer> arr) {
    if (root == null) {
      return;
    }
    for (Integer anArr : arr) {
      maxDiff = Math.max(Math.abs(anArr - root.val), maxDiff);
    }
    arr.add(root.val);
    maxAncestorDiff(root.left, arr);
    maxAncestorDiff(root.right, arr);
    arr.remove(arr.size() - 1);
  }


//  private int minWidth = Integer.MAX_VALUE, maxWidth = Integer.MIN_VALUE;

  public List<List<Integer>> verticalOrder(TreeNode root) {
    Map<Integer, List<Pair<Integer, Integer>>> map = new HashMap<>();
    verticalOrder(root, 0, 0, map);
    List<List<Integer>> ans = new ArrayList<>();
    for (int i = minWidth; i <= maxWidth; i++) {
      List<Pair<Integer, Integer>> value = map.get(i);
      value.sort(new Comparator<Pair<Integer, Integer>>() {
        @Override
        public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
          return o1.second - o2.second;
        }
      });
      List<Integer> toAdd = new ArrayList<>();
      for (Pair<Integer, Integer> p : value) {
        toAdd.add(p.first);
      }
      ans.add(toAdd);
    }
    return ans;
  }

  private void verticalOrder(TreeNode root, int width, int depth, Map<Integer, List<Pair<Integer, Integer>>> res) {
    if (root == null) {
      return;
    }
    minWidth = Math.min(minWidth, width);
    maxWidth = Math.max(maxWidth, width);
    if (!res.containsKey(width)) {
      res.put(width, new ArrayList<>());
    }
    res.get(width).add(new Pair<>(root.val, depth));
    verticalOrder(root.left, width - 1, depth + 1, res);
    verticalOrder(root.right, width + 1, depth + 1, res);
  }

  int ans;

  public int sumRootToLeafII(TreeNode root) {
    sumRootToLeaf(root, 0);
    return ans;
  }

  private void sumRootToLeaf(TreeNode root, int val) {
    val = val * 2 + root.val;
    if (root.left == null && root.right == null) {
      ans += val;
      return;
    }
    if (root.left != null) {
      sumRootToLeaf(root.left, val);
    }
    if (root.right != null) {
      sumRootToLeaf(root.right, val);
    }
  }


  public int sumRootToLeaf(TreeNode root) {
    ans = 0;
    if (root == null) {
      return 0;
    }
    sumRootToLeaf(root, "");
    return ans;
  }

  public void sumRootToLeaf(TreeNode root, String s) {
    if (root.left == null && root.right == null) {
      String s1 = s + root.val;
      int num = 0;
      for (int i = s1.length() - 1, idx = 0; i >= 0; i--, idx++) {
        if (s1.charAt(i) == 1) {
          num += Math.pow(2, idx);
        }
      }
      ans += num;
      return;
    }
    if (root.left != null) {
      sumRootToLeaf(root.left, s + root.val);
    }
    if (root.right != null) {
      sumRootToLeaf(root.right, s + root.val);
    }
  }

  public TreeNode deleteNode(TreeNode root, int key) {
    if (root == null) return root;
    if (key < root.val) {
      root.left = deleteNode(root.left, key);
    } else if (key > root.val) {
      root.right = deleteNode(root.right, key);
    } else {
      if (root.left == null) return root.right;
      if (root.right == null) return root.left;
      TreeNode newRoot = root.right, par = null;
      while (newRoot.right != null) {
        par = newRoot;
        newRoot = newRoot.left;
      }
      if (par == null) {
        newRoot.left = root.left;
        return newRoot;
      }
      par.left = newRoot.right;
      newRoot.left = root.left;
      newRoot.right = root.right;
      return newRoot;
    }
    return root;
  }

  public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
    TreeNode root;
    if (t1 == null && t2 == null) {
      root = null;
    } else if (t1 != null && t2 != null) {
      root = new TreeNode(t1.val + t2.val);
      root.left = mergeTrees(t1.left, t2.left);
      root.right = mergeTrees(t1.right, t2.right);
    } else if (t1 == null && t2 != null) {
      root = new TreeNode(t2.val);
      root.left = t2.left;
      root.right = t2.right;
    } else {
      root = new TreeNode(t1.val);
      root.left = t1.left;
      root.right = t1.right;
    }
    return root;
  }

  private int pathCount = 0;
  private Map<Integer, Integer> m;

  public int pathSum3(TreeNode root, int sum) {
    m1 = new HashMap<>();
    m1.put(0, 1);
    if (root == null) {
      return 0;
    }
    pathSum3Util(root, 0, sum);
    return pathCount;
  }

  private void pathSum3Util(TreeNode root, int sum, int requiredSum) {
    if (root == null) {
      return;
    }
    int curSum = sum + root.val;
    if (map.containsKey(curSum - requiredSum)) {
      pathCount++;
    }
    map.put(curSum, map.getOrDefault(curSum, 0) + 1);
    pathSum3Util(root.left, curSum, requiredSum);
    pathSum3Util(root.right, curSum, requiredSum);
    map.put(curSum, map.getOrDefault(curSum, 0) - 1);
    if (map.get(curSum) == 0) {
      map.remove(curSum);
    }
  }

  public boolean hasPathSum2(TreeNode root, int sum) {
    if (root == null) {
      return false;
    }
    if (root.left == null && root.right == null && root.val == sum) {
      return true;
    }
    boolean l, r;
    l = hasPathSum2(root.left, sum - root.val);
    r = hasPathSum2(root.right, sum - root.val);
    return l || r;
  }

  private int maxUniValuelength = 1;

  public int longestUnivaluePath(TreeNode root) {
    traverse3(root);
    return maxUniValuelength;
  }

  private int traverse3(TreeNode root) {
    if (root == null) {
      return 0;
    }
    int l = traverse3(root.left);
    int r = traverse3(root.right);
    int curPath = 0;
    int pathFromLeft = 0, pathFromRight = 0;
    if (satisfiesFunction(root, root.left)) {
      curPath += (l + 1);
      pathFromLeft = 1 + l;
    }
    if (satisfiesFunction(root, root.right)) {
      curPath += (r + 1);
      pathFromRight = 1 + r;
    }
    if (curPath > maxUniValuelength) {
      maxUniValuelength = curPath;
    }
    return Math.max(pathFromLeft, pathFromRight);
  }

  private boolean satisfiesFunction(TreeNode root, TreeNode left) {
    return left != null && root.val == left.val;
  }


  public boolean isSubtree(TreeNode s, TreeNode t) {
    if (areTreesEqual(s, t)) {
      return true;
    }
    if (s == null) {
      return false;
    }
    boolean l = false;
    if (s.left != null) {
      l = isSubtree(s.left, t);
    }
    boolean r = false;
    if (s.right != null) {
      r = isSubtree(s.right, t);
    }
    return l || r;
  }

  private boolean areTreesEqual(TreeNode s, TreeNode t) {
    if (s == null && t == null) {
      return true;
    }
    if (s == null && t != null) {
      return false;
    }
    if (s != null && t == null) {
      return false;
    }
    if (s.val != t.val) {
      return false;
    }
    return areTreesEqual(s.left, t.left) && areTreesEqual(s.right, t.right);
  }

  public TreeNode sortedListToBST(ListNode a) {
    if (a == null) {
      return null;
    }
    if (a.next == null) {
      return new TreeNode(a.val);
    }
    ListNode slow = a, fast = a.next;
    while (fast != null) {
      if (fast.next == null || fast.next.next == null) {
        break;
      }
      slow = slow.next;
      fast = fast.next.next;
    }
    ListNode rootListNode = slow.next;
    slow.next = null;
    TreeNode root = new TreeNode(rootListNode.val);
    root.left = sortedListToBST(a);
    root.right = sortedListToBST(rootListNode.next);
    return root;
  }

  int maxSum = Integer.MIN_VALUE;

  public int maxPathSum(TreeNode A) {
    if (A == null) {
      return 0;
    }
    traverse2(A);
    return maxSum;
  }

  private int traverse2(TreeNode A) {
    if (A == null) {
      return 0;
    }
    int l = traverse2(A.left);
    int r = traverse2(A.right);
    int curSum = A.val + l + r;
    if (curSum < 0) {
      curSum = A.val;
    }
    if (curSum > maxSum) {
      maxSum = curSum;
    }
    int valueToReturn = A.val + Math.max(l, r);
    return (valueToReturn >= 0) ? valueToReturn : 0;
  }

  public int minCameraCover(TreeNode root) {
    int[] ans = traverse(root);
    return Math.min(ans[1], ans[2]);
  }

  private int[] traverse(TreeNode root) {
    if (root == null) {
      return new int[]{0, 0, 99999};
    }
    int[] left = traverse(root.left);
    int[] right = traverse(root.right);
    int d0 = left[1] + right[1];
    int d1 = Math.min(Math.min(left[1], left[2]) + right[2], Math.min(left[2], Math.min(right[1], right[2])));
    int d2 = 1 + Math.min(left[0], Math.min(left[1], left[2])) + Math.min(right[0], Math.min(right[1], right[2]));
    return new int[]{d0, d1, d2};
  }

  private int minDiff;
  private int prev;

  public int getMinimumDifference(TreeNode root) {
    minDiff = Integer.MAX_VALUE;
    prev = -1;
    inOrder(root);
    return minDiff;
  }

  private void inOrder(TreeNode root) {
    if (root == null) {
      return;
    }
    inOrder(root.left);
    if (prev != -1) {
      minDiff = Math.min(minDiff, root.val - prev);
    }
    prev = root.val;
    inOrder(root.right);
  }

  private int ans2 = 0;

  public int distributeCoins(TreeNode root) {
    ans2 = 0;
    distributeCoinsUtil(root);
    return ans2;
  }

  private int distributeCoinsUtil(TreeNode root) {
    if (root == null) return 0;
    int l = distributeCoins(root.left);
    int r = distributeCoins(root.right);
    ans2 += Math.abs(l) + Math.abs(r);
    return l + r + root.val - 1;
  }

  private boolean shouldCount;
  private int sum;

  public int rangeSumBST(TreeNode root, int L, int R) {
    shouldCount = false;
    sum = 0;
    inorder(root, L, R);
    return sum;
  }

  private void inorder(TreeNode root, int l, int r) {
    if (root == null) return;
    inorder(root.left, l, r);
    if (root.val == l) shouldCount = true;
    if (shouldCount) {
      sum += root.val;
    }
    if (root.val == r) shouldCount = false;
    inorder(root.right, l, r);
  }


  private int maxCount;
  private Map<Integer, Integer> m1;

  public int[] findMode(TreeNode root) {
    if (root == null) {
      return new int[]{};
    }
    maxCount = 0;
    m1 = new HashMap<>();
    calc(root);
    return findModeUtil(root).stream().mapToInt(Integer::intValue).toArray();

  }

  private Set<Integer> findModeUtil(TreeNode root) {
    if (root == null) {
      return new HashSet<>();
    }
    Set<Integer> ans = new HashSet<>();
    if (m1.get(root.val) == maxCount) {
      ans.add(root.val);
    }
    Set<Integer> left = findModeUtil(root.left);
    Set<Integer> right = findModeUtil(root.right);
    ans.addAll(left);
    ans.addAll(right);
    return ans;
  }

  private void calc(TreeNode root) {
    if (root == null) {
      return;
    }
    if (!m1.containsKey(root.val)) {
      m1.put(root.val, 0);
    }
    m1.put(root.val, m1.get(root.val) + 1);
    maxCount = Math.max(maxCount, m1.get(root.val));
    calc(root.left);
    calc(root.right);
  }


  private Map<Integer, Integer> map = new HashMap<>();

  public int[] findFrequentTreeSum(TreeNode root) {
    sumUtil(root);
    ArrayList<Integer> ans = new ArrayList<>();
    for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
      if (entry.getValue() == maxCount) {
        ans.add(entry.getValue());
      }
    }
    return ans.stream().mapToInt(Integer::intValue).toArray();
//      return map.entrySet()
//              .stream()
//              .filter(x -> x.getValue() == maxCount)
//              .map(Map.Entry::getKey)
//              .mapToInt(Integer::intValue).toArray();
  }


  private int sumUtil(TreeNode root) {
    if (root == null) {
      return 0;
    }
    int l = sumUtil(root.left);
    int r = sumUtil(root.right);
    int sum = l + r + root.val;
    if (!map.containsKey(sum)) {
      map.put(sum, 0);
    }
    int newCount = map.get(sum) + 1;
    map.put(sum, newCount);
    if (newCount > maxCount) {
      maxCount = newCount;
    }
    return sum;
  }

  private static int lca2(TreeNode T, int v, int w) {
    ArrayList<Integer> pathToV = new ArrayList<>();
    ArrayList<Integer> pathToW = new ArrayList<>();
    find(T, v, pathToV);
    find(T, w, pathToW);
    if (pathToV.size() == 0 || pathToW.size() == 0) {
      return -1;
    }
    int c = 1;
    for (; c < Math.min(pathToV.size(), pathToW.size()); c++) {
      if (!pathToV.get(c).equals(pathToW.get(c))) {
        return pathToV.get(c - 1);
      }
    }
    return pathToV.get(c - 1);
  }

  private static boolean find(TreeNode T, int v, ArrayList<Integer> path) {
    if (T == null) {
      return false;
    }
    path.add(T.val);
    if (T.val == v) {
      return true;
    }
    if (find(T.left, v, path) || find(T.right, v, path)) {
      return true;
    }
    path.remove(new Integer(T.val));
    return false;
  }

  private static int lca(TreeNode T, int v, int w) {
    if (T == null) {
      return -1;
    }
    if (T.val == v && T.val == w) {
      return T.val;
    }
    if (T.val == v) {
      if (findElement(T.left, w) || findElement(T.right, w)) {
        return T.val;
      }
    } else if (T.val == w) {
      if (findElement(T.left, v) || findElement(T.right, v)) {
        return T.val;
      }
    } else {
      boolean lv = findElement(T.left, v);
      boolean lw = findElement(T.left, w);
      if (lv && lw) {
        return lca(T.left, v, w);
      }
      if (!lv && !lw) {
        return lca(T.right, v, w);
      }
      if (lv) {
        boolean rw = findElement(T.right, w);
        if (rw) {
          return T.val;
        } else {
          return -1;
        }
      } else if (lw) {
        boolean rv = findElement(T.right, v);
        if (rv) {
          return T.val;
        } else {
          return -1;
        }
      }
    }
    return -1;
  }

  private static boolean findElement(TreeNode T, int v) {
    if (T == null) {
      return false;
    }
    return (T.val == v) || findElement(T.left, v) || findElement(T.right, v);
  }

  public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode A) {
    ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
    Queue<TreeNode> q = new LinkedList<TreeNode>();
    if (A != null) {
      q.add(A);
    }
    int c = 1;
    boolean flag = false;
    while (!q.isEmpty()) {
      ArrayList<Integer> temp = new ArrayList<>();
      int newCount = 0;
      while (c > 0) {
        TreeNode t = q.poll();
        temp.add(t.val);
        c--;
        if (t.left != null) {
          q.add(t.left);
          newCount++;
        }
        if (t.right != null) {
          q.add(t.right);
          newCount++;
        }
      }
      c = newCount;
      if (flag) {
        Collections.reverse(temp);
      }
      ans.add(temp);
      flag = !flag;
    }
    return ans;
  }

  public int isSymmetric(TreeNode A) {
    TreeNode B = A;
    return isSymmetricUtil(A, B) ? 1 : 0;
  }

  private boolean isSymmetricUtil(TreeNode A, TreeNode B) {
    if (A == null && B == null) {
      return true;
    }
    if ((A == null && B != null) || (A != null && B == null)) {
      return false;
    }
    boolean l = isSymmetricUtil(A.left, B.right);
    boolean r = isSymmetricUtil(A.right, B.left);
    return l && r && A.val == B.val;
  }

  public int hasPathSum(TreeNode A, int B) {
    if (A == null) {
      return (B == 0) ? 1 : 0;
    }
    if ((A.left == null && A.right == null) && A.val == B) {
      return 1;
    }
    int l = hasPathSum(A.left, B - A.val);
    int r = hasPathSum(A.right, B - A.val);
    return (l == 1 || r == 1) ? 1 : 0;
  }

  int c;

  public int kthsmallest(TreeNode A, int B) {
    c = B;
    ans2 = -1;
    kthsmallestUtil(A);
    return ans2;
  }

  private void kthsmallestUtil(TreeNode a) {
    if (a == null) {
      return;
    }
    kthsmallestUtil(a.left);
    c--;
    if (c == 0) {
      ans2 = a.val;
      return;
    }
    kthsmallestUtil(a.right);
  }

  public TreeNode invertTree(TreeNode A) {
    if (A == null || (A.left == null && A.right == null)) {
      return A;
    }

    if (A.left == null) {
      A.left = A.right;
      A.right = null;
    } else if (A.right == null) {
      A.right = A.left;
      A.left = null;
    } else {
      TreeNode temp = A.left;
      A.left = A.right;
      A.right = temp;
    }
    invertTree(A.left);
    invertTree(A.right);
    return A;
  }

  private int isBalanced = 1;

  public int isBalanced(TreeNode A) {
    isBalanced = 1;
    calcHeight(A);
    return isBalanced;
  }

  private int calcHeight(TreeNode t) {
    if (t == null) {
      return 0;
    }
    int l = calcHeight(t.left);
    int r = calcHeight(t.right);
    if (Math.abs(l - r) > 1) {
      isBalanced = 0;
    }
    return Math.max(l, r) + 1;
  }

  public static List<Integer> solve(String A, List<String> B) {
    String[] goodWords = A.split("_");
    Trie t = new Trie();
    for (String goodWord : goodWords) {
      t.insert(goodWord);
    }

    ArrayList<Review> reviews = new ArrayList<>();

    for (int i = 0; i < B.size(); i++) {
      String review = B.get(i);
      String[] reviewWords = review.split("_");
      int goodness = 0;
      for (String reviewWord : reviewWords) {
        if (t.search(reviewWord)) {
          goodness++;
        }
      }
      reviews.add(new Review(i, goodness));
    }

    reviews.sort((o1, o2) -> Integer.compare(o2.value, o1.value));

    return reviews.stream().map(r -> r.idx).collect(Collectors.toCollection(ArrayList::new));
  }

  static class Review {
    int idx;
    int value;

    Review(int idx, int value) {
      this.idx = idx;
      this.value = value;
    }
  }

  public boolean isValidBST(TreeNode A, Integer maxValue, Integer minValue) {
    if (A == null) {
      return true;
    }
    if (A.left != null && (A.left.val >= A.val || (minValue != null && A.left.val <= minValue))) {
      return false;
    }
    if (A.right != null && (A.right.val <= A.val || (maxValue != null && A.right.val >= maxValue))) {
      return false;
    }
    return isValidBST(A.left, A.val, minValue) && isValidBST(A.right, maxValue, A.val);
  }

}
