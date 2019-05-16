
import java.util.*;

public class Array2 {

//  public static void main(String[] args) {
//    TreeNodeWithChildren root = new TreeNodeWithChildren(6);
//    root.left = new TreeNodeWithChildren(3);
//    root.left.left = new TreeNodeWithChildren(2);
//    root.left.right = new TreeNodeWithChildren(5);
//    root.right = new TreeNodeWithChildren(7);
//    root.right.right = new TreeNodeWithChildren(9);
//    ListNode first = new ListNode(2);
////    first.next = new ListNode(1);
////    first.next.next = new ListNode(2);
////    first.next.next.next = new ListNode(3);
////    first.next.next.next.next = new ListNode(4);
//    ListNode second = new ListNode(0);
//    second.next = new ListNode(1);
//    second.next.next = new ListNode(2);
//    second.next.next.next = new ListNode(9);
//    second.next.next.next.next = new ListNode(9);
//    //System.out.println(-4 % 10);
////    System.out.println(findRank2(new StringBuilder("abcdefghijklmnopqrst").reverse().toString()));
////    System.out.println(findRank3(new StringBuilder("abcdefghijklmnopqrst").reverse().toString()));
//    //System.out.println(findRank4(new StringBuilder("abcdefghijklmnopqrst").reverse().toString()));
//
//
//
//  }

  public static ArrayList<Integer> flip(String A) {
    if (A == null || A.length() == 0) {
      return new ArrayList<>();
    }
    int l = A.length();
    int curSum = -1, L = -1, R = -1, maxSum = -1, curL = -1, curR = -1;
    for (int i = 0; i < l; i++) {
      int c = (A.charAt(i) == '1') ? -1 : 1;
      if (c == 1) {
        if (curSum < 0) {
          curSum = 0;
          curL = i;
        }
        curSum = curSum + c;
        curR = i;
        if (curSum > maxSum) {
          R = curR;
          maxSum = curSum;
          L = curL;
        }

      } else {
        curSum = curSum + c;
      }
    }
    if (L == -1) {
      return new ArrayList<>();
    }
    return new ArrayList<>(Arrays.asList(L + 1, R + 1));
  }


  void populateAndIncreaseCount(int[] count, String str) {
    // count is initialized to zero for all indices
    for (int i = 0; i < str.length(); ++i) {
      count[str.charAt(i)]++;
    }

    for (int i = 1; i < 256; ++i)
      count[i] += count[i - 1];
  }

  static final int M = 1000003;

  static int[] fs;

  static void fillFactorials(int l) {
    fs = new int[l + 1];
    fs[0] = 1;
    for (int i = 1; i <= l; i++) {
      fs[i] = (fs[i - 1] * i) % M;
    }
  }

  public static int findRank4(String a) {
    int l = a.length();
    fillFactorials(l);

    int[] chars = new int[256];
    for (int i = 0; i < a.length(); i++) {
      byte c = (byte) a.charAt(i);
      chars[c]++;
    }

    int res = 1;
    for (int i = 0; i < l; i++) {
      int lessThan = 0;
      for (int j = 0; j < (byte) a.charAt(i); j++) {
        lessThan += chars[j];
      }
      res += fs[l - i - 1] * lessThan % M;
      chars[a.charAt(i)] = 0;
    }
    return res % M;
  }

  private static int count;
  private static int mod = 1000003;

  public static int findRank2(String A) {
    List<Character> sorted = new ArrayList<>();
    for (int i = 0; i < A.length(); i++) {
      sorted.add(A.charAt(i));
    }
    Collections.sort(sorted);
    ArrayList<Long> facts = new ArrayList<>(A.length());
    facts.add(0L);
    facts.add(1L);
    for (int i = 2; i < A.length(); i++) {
      facts.add(i * facts.get(i - 1));
    }
    long count2 = 0;
    for (int i = 0; i < A.length() - 1; i++) {
      int c = 0;
      for (int j = i + 1; j < A.length(); j++) {
        if (A.charAt(j) < A.charAt(i)) {
          c++;
        }
      }
      int idx = Arrays.binarySearch(sorted.toArray(), A.charAt(i));
      count2 += ((facts.get(A.length() - i - 1)) * idx) % mod;
      count2 %= mod;
      sorted.remove(sorted.indexOf(A.charAt(i)));

    }
    return (int) count2 + 1;
  }


  public static int findRank3(String A) {
    int length = A.length();
    long strFactorial = factorial(length);
    long rank = 1;
    for (int i = 0; i < length; i++) {
      strFactorial /= length - i;
      rank += findSmallerInRight(A, i, length - 1) * strFactorial;
    }
    rank %= 1000003;
    return (int) rank;
  }

  public static long factorial(int n) {
    return n <= 1 ? 1 : (n * factorial(n - 1));
  }

  public static int findSmallerInRight(String A, int low, int high) {
    int countRight = 0;
    for (int i = low + 1; i <= high; i++) {
      if (A.charAt(i) < A.charAt(low))
        countRight++;
    }
    return countRight;
  }

  public static int findRank(String A) {

    char[] sorted = A.toCharArray();
    Arrays.sort(sorted);
    boolean visited[] = new boolean[sorted.length];
    for (int i = 0; i < visited.length; i++) {
      visited[i] = false;
    }
    char[] cur = new char[A.length()];
    for (int i = 0; i < sorted.length; i++) {
      cur[0] = sorted[i];
      visited[i] = true;
      int tmp = calc(1, sorted, visited, A, cur);
      if (tmp != -1) {
        return tmp;
      }
      visited[i] = false;
    }
    return -1;
  }

  private static int calc(int idx, char[] sorted, boolean[] visited, String A, char[] cur) {
    if (idx == A.length()) {
      count = (count + 1) % mod;
      if (Arrays.equals(A.toCharArray(), cur)) {
        return count;
      }
      return -1;
    }
    for (int i = 0; i < sorted.length; i++) {
      if (!visited[i]) {
        visited[i] = true;
        cur[idx] = sorted[i];
        int tmp = calc(idx + 1, sorted, visited, A, cur);
        if (tmp != -1) {
          return tmp;
        }
        visited[i] = false;
      }
    }
    return -1;
  }

  public static int nTriang(List<Integer> A) {
    long mod = 1000000000 + 7;
    long count = 0;
    if (A == null) {
      return (int) count;
    }
    Collections.sort(A);
    for (int i = A.size() - 1; i > 1; i--) {
      int idx = 0;
      for (int j = i - 1; j > 0; j--) {
        //bin search for smallest element which is greater than A.get(i)-A.get(j)
        idx = binSearch(A.get(i) - A.get(j) + 1, idx, j - 1, A);
        if (idx != -1) {
          count += j - idx;
          count %= mod;
        } else {
          break;
        }
      }
    }
    return (int) count;
  }

  private static int binSearch(int target, int begin, int end, List<Integer> A) {
    if (begin > end) {
      return -1;
    }
    if (begin == end) {
      return A.get(begin) >= target ? begin : -1;
    }
    int mid = (end + begin) / 2;
    if (A.get(mid) >= target) {
      //go left but don't loose this element
      end = mid;
    } else if (A.get(mid) < target) {
      begin = mid + 1;
    }
    return binSearch(target, begin, end, A);
  }

  private boolean satisy(int i, int j, int k) {
    return i + j > k && i + k > j && k + j > i;
  }

  public int Mod(int A, int B, int C) {
    if (B == 0) {
      return 1 % C;
    }
    if (B == 1 || A == 1) {
      return Math.floorMod(A, C);
    }
    long val = Mod(A, B / 2, C);
    if (B % 2 == 0) {
      return (int) Math.floorMod(val * val, C);
    } else {
      return (int) ((Math.floorMod(val * val, C) * Math.floorMod(A, C)) % C);
    }
  }

  public static String minWindow(String A, String B) {
    if (B == null || B.length() == 0) {
      return "";
    }
    Set<Character> charSet = new HashSet<>();
    Map<Character, Integer> countMap = new HashMap<>();
    for (Character c : B.toCharArray()) {
      charSet.add(c);
      if (countMap.containsKey(c)) {
        countMap.put(c, countMap.get(c) + 1);
      } else {
        countMap.put(c, 1);
      }
    }
    int start = 0, end = 0;
    int minStart = -1, minLength = Integer.MAX_VALUE;
    while (end < A.length()) {
      char endChar = A.charAt(end);
      if (charSet.contains(endChar)) {
        countMap.put(endChar, countMap.get(endChar) - 1);
      }
      if (satisfies(countMap)) {
        int curLength = end - start + 1;
        if (curLength < minLength) {
          minLength = curLength;
          minStart = start;
        }
        start++;
        char startChar = A.charAt(start - 1);
        if (countMap.containsKey(startChar)) {
          countMap.put(startChar, countMap.get(startChar) + 1);
        }
        while (satisfies(countMap) && start <= end) {
          curLength = end - start + 1;
          if (curLength < minLength) {
            minLength = curLength;
            minStart = start;
          }
          start++;
          startChar = A.charAt(start - 1);
          if (countMap.containsKey(startChar)) {
            countMap.put(startChar, countMap.get(startChar) + 1);
          }
        }
      }
      end++;
    }
    if (minLength == Integer.MAX_VALUE) {
      return "";
    }
    return A.substring(minStart, minStart + minLength);
  }

  private static boolean satisfies(Map<Character, Integer> countSet) {
    Set<Character> keySet = countSet.keySet();
    for (Character c : keySet) {
      if (countSet.get(c) > 0) {
        return false;
      }
    }
    return true;
  }


  static ArrayList<ArrayList<String>> solutions;

  public static ArrayList<ArrayList<String>> solveNQueens(int a) {
    solutions = new ArrayList<>();
    if (a == 0) {
      return solutions;
    }
    calc(0, new ArrayList<>(), a);
    return solutions;
  }

  private static void calc(int idx, ArrayList<String> curSol, int size) {
    if (idx == size) {
      ArrayList<String> copy = new ArrayList<>();
      copy.addAll(curSol);
      solutions.add(copy);
      return;
    }
    char[] curArr = getDefaultString(size);
    for (int i = 0; i < size; i++) {
      if (isValid(idx, i, curSol)) {
        curArr[i] = 'Q';
        curSol.add(new String(curArr));
        calc(idx + 1, curSol, size);
        curSol.remove(curSol.size() - 1);
        curArr[i] = '.';
      }
    }
  }

  private static char[] getDefaultString(int size) {
    char[] curArr = new char[size];
    for (int i = 0; i < size; i++) {
      curArr[i] = '.';
    }
    return curArr;
  }

  private static boolean isValid(int idx, int pos, ArrayList<String> curSol) {
    for (int i = 0; i < curSol.size(); i++) {
      String curStr = curSol.get(i);
      for (int j = 0; j < curStr.length(); j++) {
        if (curStr.charAt(j) == 'Q' && matches(i, j, idx, pos)) {
          return false;
        }
      }
    }
    return true;
  }

  private static boolean matches(int i, int j, int idx, int pos) {
    return (i == idx || j == pos || Math.abs(idx - i) == Math.abs(pos - j));
  }

  public static int maxSpecialProduct(List<Integer> A) {
    if (A == null || A.size() == 0) {
      return 0;
    }
    Integer size = A.size();
    ArrayList<Integer> rightSpecial = new ArrayList<>(size);
    Stack<Integer> st = new Stack<>();
    for (Integer i = size - 1; i >= 0; i--) {
      while (!st.empty() && A.get(st.peek()) <= A.get(i)) {
        st.pop();
      }
      if (st.empty()) {
        rightSpecial.add(0);
      } else {
        rightSpecial.add(st.peek());
      }
      st.push(i);
    }
    Collections.reverse(rightSpecial);

    st.clear();
    ArrayList<Integer> leftSpecial = new ArrayList<>(size);
    for (int i = 0; i < size; i++) {
      while (!st.empty() && A.get(st.peek()) <= A.get(i)) {
        st.pop();
      }
      if (st.empty()) {
        leftSpecial.add(0);
      } else {
        leftSpecial.add(st.peek());
      }
      st.push(i);
    }
    Integer maxProd = 0;
    long MODULO = 1000000007;
    for (Integer i = 0; i < size; i++) {
      int curProd = (int) (((long) (leftSpecial.get(i)) * (long) rightSpecial.get(i)) % MODULO);
      if (curProd > maxProd) {
        maxProd = curProd;
      }
    }
    return maxProd;
  }

  public ArrayList<Integer> nextGreater(ArrayList<Integer> A) {
    if (A == null) {
      return new ArrayList<>();
    }
    int size = A.size();
    ArrayList<Integer> ans = new ArrayList<>(size);
    Stack<Integer> st = new Stack<>();
    for (int i = A.size() - 1; i >= 0; i--) {
      while (!st.empty() && st.peek() <= A.get(i)) {
        st.pop();
      }
      if (st.empty()) {
        ans.add(-1);
      } else {
        ans.add(st.peek());
      }
      st.push(A.get(i));
    }
    Collections.reverse(ans);
    return ans;
  }

  public int maxArea(ArrayList<Integer> A) {
    if (A == null) {
      return 0;
    }
    int maxArea = 0;
    int l = 0, r = A.size() - 1;
    while (l < r) {
      int tempArea = Math.min(A.get(l), A.get(r)) * (r - l);
      maxArea = Math.max(tempArea, maxArea);
      if (Math.min(A.get(l), A.get(r)) == A.get(l)) {
        l++;
      } else {
        r--;
      }
    }
    return maxArea;
  }

  public int maxArea2(ArrayList<Integer> A) {
    if (A == null || A.size() < 1) {
      return 0;
    }
    int size = A.size();
    int maxArea = Integer.MIN_VALUE;
    for (int i = 0; i < size; i++) {
      for (int j = i + 1; j < size; j++) {
        int tempArea = Math.min(A.get(i), A.get(j)) * (j - i);
        if (tempArea > maxArea) {
          maxArea = tempArea;
        }
      }
    }
    return maxArea;
  }


  public static int countInversions(List<Integer> A) {
    return merge(A, 0, A.size() - 1);
  }

  private static int merge(List<Integer> A, int l, int h) {
    int inv = 0;
    if (l < h) {
      int mid = (l + h) / 2;
      inv += merge(A, l, mid);
      inv += merge(A, mid + 1, h);
      inv += merge(A, l, mid, mid + 1, h);
    }
    return inv;
  }

  public static int merge(List<Integer> A, int l1, int h1, int l2, int h2) {
    int i = l1, j = l2, k = l1;
    int inv = 0;
    ArrayList<Integer> temp = new ArrayList<>();
    while (i <= h1 && j <= h2) {
      if (A.get(i) <= A.get(j)) {
        temp.add(A.get(i));
        i++;
        k++;
      } else {
        temp.add(A.get(j));
        j++;
        k++;
        inv += (h1 - i + 1);
      }
    }
    while (i <= h1) {
      temp.add(A.get(i));
      k++;
      i++;
    }
    while (j <= h2) {
      temp.add(A.get(j));
      k++;
      j++;
    }
    j = 0;
    i = l1;
    while (j < temp.size()) {
      A.set(i++, temp.get(j++));
    }
    return inv;
  }

  public static int adjacent(ArrayList<List<Integer>> A) {
    int size = A.get(0).size();
    ArrayList<Integer> arr = new ArrayList<>();
    for (int i = 0; i < size; i++) {
      arr.add(Integer.max(A.get(0).get(i), A.get(1).get(i)));
    }
    int dp[][] = new int[2][size];
    dp[0][0] = 0;
    dp[1][0] = arr.get(0);
    for (int i = 1; i < size; i++) {
      dp[1][i] = arr.get(i) + dp[0][i - 1];
      dp[0][i] = Integer.max(dp[0][i - 1], dp[1][i - 1]);
    }
    return Integer.max(dp[0][size - 1], dp[1][size - 1]);
  }


  public static TreeNode buildTree(List<Integer> inOrder, List<Integer> postOrder) {
    int l = postOrder.size();
    int rootVal = postOrder.get(l - 1);
    TreeNode root = new TreeNode(rootVal);
    if (l == 1) {
      return root;
    }
    int idx = 0;
    for (; idx < l; idx++) {
      if (inOrder.get(idx) == rootVal) {
        break;
      }
    }
    ArrayList<Integer> leftIn = new ArrayList<>();
    ArrayList<Integer> leftPost = new ArrayList<>();
    for (int i = 0; i < idx; i++) {
      leftIn.add(inOrder.get(i));
      leftPost.add(postOrder.get(i));
    }
    ArrayList<Integer> rightIn = new ArrayList<>();
    ArrayList<Integer> rightPost = new ArrayList<>();
    for (int i = idx + 1; i < l; i++) {
      rightIn.add(inOrder.get(i));
      rightPost.add(postOrder.get(i - 1));
    }
    root.left = buildTree(leftIn, leftPost);
    root.right = buildTree(rightIn, rightPost);
    return root;
  }

  public static String simplifyPath(String A) {
    String[] dirs = A.split("/");
    Stack<String> st = new Stack<>();
    for (int i = 0; i < dirs.length; i++) {
      switch (dirs[i]) {
        case ".":
          break;
        case "..":
          if (!st.empty()) {
            st.pop();
          }
          break;
        default:
          if (dirs[i].length() != 0) {
            StringBuilder sb = new StringBuilder(dirs[i]);
            st.push(sb.reverse().toString());
          }
      }
    }
    StringBuilder stringBuilder = new StringBuilder();
    if (st.empty()) {
      stringBuilder.append("/");
    } else {
      while (!st.empty()) {
        stringBuilder.append(st.pop());
        stringBuilder.append("/");
      }
    }
    return stringBuilder.reverse().toString();
  }

  public static void sortColors(List<Integer> a) {
    if (a == null) {
      return;
    }
    int zerodx = -1, oneidx = -1, twoidx = -1, curidx = 0;
    while (curidx < a.size()) {
      if (a.get(curidx) == 0) {
        a.set(++twoidx, 2);
        a.set(++oneidx, 1);
        a.set(++zerodx, 0);
      } else if (a.get(curidx) == 1) {
        a.set(++twoidx, 2);
        a.set(++oneidx, 1);
      } else if (a.get(curidx) == 2) {
        a.set(++twoidx, 2);
      }
      curidx++;
    }
    System.out.println(a);
  }

  public static ArrayList<Integer> preorderTraversal(TreeNode A) {
    ArrayList<Integer> ans = new ArrayList<>();
    if (A == null) {
      return ans;
    }
    Stack<TreeNode> st = new Stack<>();
    st.push(A);
    TreeNode temp;
    while (!st.empty()) {
      temp = st.pop();
      ans.add(temp.val);
      if (temp.right != null) {
        st.push(temp.right);
      }
      if (temp.left != null) {
        st.push(temp.left);
      }
    }
    return ans;
  }

  public static ArrayList<Integer> lszero(List<Integer> A) {
    if (A == null || A.size() == 0) {
      return new ArrayList<>();
    }
    ArrayList<Main.IntegerPair> sumIdx = new ArrayList<>(A.size());
    int sum = 0;
    sumIdx.add(new Main.IntegerPair(0, -1));
    for (int i = 0; i < A.size(); i++) {
      sum += A.get(i);
      sumIdx.add(new Main.IntegerPair(sum, i));
    }
    sumIdx.sort(Comparator.comparing(o -> o.first));
    int ansStart = Integer.MAX_VALUE, ansEnd = Integer.MAX_VALUE, ansLength = -1;
    ArrayList<Integer> ans = new ArrayList<>();
    int idx = 0;
    while (idx < sumIdx.size()) {
      int jdx = idx + 1;
      while (jdx != sumIdx.size() && sumIdx.get(idx).first.equals(sumIdx.get(jdx).first)) {
        jdx++;
      }
      if (jdx - 1 != idx) {
        int curLength;
        if (sumIdx.get(idx).second == -1) {
          curLength = sumIdx.get(jdx - 1).second + 1;
        } else {
          curLength = sumIdx.get(jdx - 1).second - sumIdx.get(idx).second;
        }
        if (curLength > ansLength || (curLength == ansLength && sumIdx.get(idx).second + 1 < ansStart)) {
          ansLength = curLength;
          ansStart = sumIdx.get(idx).second + 1;
          ansEnd = sumIdx.get(jdx - 1).second;
        }
      }
      idx = jdx;
    }
    int i = 0;
    while (i < ansLength) {
      ans.add(A.get(ansStart + i));
      i++;
    }
    return ans;
  }


  public static ArrayList<Integer> rodCut(int A, List<Integer> B) {
    ArrayList<Integer> ans = new ArrayList<>();
    B.sort(Comparator.naturalOrder());
    calc(0, A, B, ans);
    return ans;
  }

  private static int calc(int start, int end, List<Integer> cuts, ArrayList<Integer> curArray) {
//    System.out.println("start " + start + " end: " + end);
    if (cuts.size() == 1) {
      curArray.add(cuts.get(0));
      return (end - start);
    }

    int sumMin = Integer.MAX_VALUE;
    ArrayList<Integer> sumMinArray = null;

    for (int i = 0; i < cuts.size(); i++) {
      int cut = cuts.get(i);

      ArrayList<Integer> leftCuts = new ArrayList<>();
      for (int j = 0; j < i; j++) {
        leftCuts.add(cuts.get(j));
      }

      ArrayList<Integer> rightCuts = new ArrayList<>();
      for (int j = i + 1; j < cuts.size(); j++) {
        rightCuts.add(cuts.get(j));
      }

      ArrayList<Integer> leftArray = new ArrayList<>(), rightArray = new ArrayList<>();
      int leftSum = 0, rightSum = 0;
      if (leftCuts.size() != 0) {
        leftSum = calc(start, cut, leftCuts, leftArray);
      }
      if (rightCuts.size() != 0) {
        rightSum = calc(cut, end, rightCuts, rightArray);
      }

      int totalSum = leftSum + (end - start) + rightSum;
      ArrayList<Integer> newMinArray = new ArrayList<>();
      newMinArray.add(cut);
      newMinArray.addAll(leftArray);
      newMinArray.addAll(rightArray);
      if (totalSum <= sumMin) {
        if (sumMinArray == null) {
          sumMinArray = new ArrayList<>();
          sumMinArray.addAll(newMinArray);
          sumMin = totalSum;
        } else {
          //if lexicographically smaller, replace
          if (isSmaller(newMinArray, sumMinArray)) {
            sumMinArray.clear();
            sumMin = totalSum;
            sumMinArray.addAll(newMinArray);
          }
        }
      }
//      if (newMinArray.size() == 3) {
//        System.out.println("totalSum: " + totalSum);
//        System.out.println("array: " + newMinArray.toString());
//        System.out.println("leftarray: " + leftArray.toString());
//        System.out.println("rightarray: " + rightArray.toString());
//      }
    }
    curArray.addAll(sumMinArray);
    return sumMin;
  }

  private static boolean isSmaller(ArrayList<Integer> o1, ArrayList<Integer> o2) {
    int i = 0, j = 0;
    while (i < o1.size() && j < o2.size()) {
      if (o1.get(i) < o2.get(j)) {
        return true;
      } else if (o1.get(i) > o2.get(j)) {
        return true;
      } else {
        i++;
        j++;
      }
    }
    if (i < o1.size()) {
      return false;
    }
    return true;
  }

  public static ArrayList<ArrayList<Integer>> fourSum(List<Integer> A, int B) {
    Set<ArrayList<Integer>> s = new HashSet<>();
    ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
    if (A == null || A.size() == 0) {
      return ans;
    }
    A.sort(Comparator.naturalOrder());
    int start, end, target;
    for (int i = 0; i < A.size(); i++) {
      for (int j = i + 1; j < A.size(); j++) {
        start = j + 1;
        end = A.size() - 1;
        target = B - (A.get(i) + A.get(j));
        while (start < end) {
          int curSum = A.get(start) + A.get(end);
          if (curSum == target) {
            s.add(getList(A, start, end, i, j));
            start++;
            end--;
          } else if (curSum > target) {
            end--;
          } else {
            start++;
          }
        }
      }
    }
    ans.addAll(s);
    ans.sort((o1, o2) -> {
      int i = 0, j = 0;
      while (i < o1.size()) {
        if (o1.get(i) < o2.get(j)) {
          return -1;
        } else if (o1.get(i) > o2.get(j)) {
          return 1;
        }
        i++;
        j++;
      }
      return 0;
    });
    return ans;
  }

  private static ArrayList<Integer> getList(List<Integer> A, int start, int end, int i, int j) {
    ArrayList<Integer> arr = new ArrayList<>();
    arr.add(A.get(i));
    arr.add(A.get(j));
    arr.add(A.get(start));
    arr.add(A.get(end));
    return arr;
  }

  private static void print(ListNode b) {
    while (b != null) {
      System.out.print(b.val);
      b = b.next;
    }
    System.out.println();
  }

  public static ListNode addTwoNumbersPractice(ListNode A, ListNode B) {
    int l1 = size(A), l2 = size(B);
    ListNode smaller, larger;
    if (l1 > l2) {
      larger = A;
      smaller = B;
    } else {
      larger = B;
      smaller = A;
    }
    ListNode res = larger;
    int c = 0, curSum;
    while (smaller.next != null) {
      curSum = smaller.val + larger.val + c;
      larger.val = curSum % 10;
      c = curSum / 10;
      smaller = smaller.next;
      larger = larger.next;
    }
    curSum = smaller.val + larger.val + c;
    larger.val = curSum % 10;
    c = curSum / 10;
    while (c != 0) {
      if (larger.next == null) {
        larger.next = new ListNode(c);
        c = 0;
      } else {
        larger = larger.next;
        curSum = larger.val + c;
        larger.val = curSum % 10;
        c = curSum / 10;
      }
    }
    return res;
  }

  private static int size(ListNode b) {
    int c = 0;
    while (b != null) {
      c++;
      b = b.next;
    }
    return c;
  }

  public static ListNode addTwoNumbers(ListNode A, ListNode B) {
    int c = 0, cursum;
    if (A == null) {
      return B;
    }
    if (B == null) {
      return A;
    }
    int l1 = size(A), l2 = size(B);
    System.out.println(l1);
    System.out.println(l2);
    ListNode larger, smaller;
    if (l1 > l2) {
      larger = A;
      smaller = B;
    } else {
      larger = B;
      smaller = A;
    }
    ListNode ans = larger;
    while (smaller.next != null) {
      cursum = c + larger.val + smaller.val;
      c = cursum / 10;
      larger.val = cursum % 10;
      smaller = smaller.next;
      larger = larger.next;
    }
    cursum = smaller.val + c + larger.val;
    larger.val = cursum % 10;
    c = cursum / 10;
    while (c != 0) {
      if (larger.next == null) {
        larger.next = new ListNode(c);
        c = 0;
      } else {
        larger = larger.next;
        cursum = larger.val + c;
        larger.val = cursum % 10;
        c = cursum / 10;
      }
    }
    return ans;
  }

  public static ListNode reorderList(ListNode A) {
    if (A == null || A.next == null) {
      return A;
    }
    //find centre
    ListNode slow = A;
    ListNode fast = A.next;
    ListNode center;
    while (fast.next != null && fast.next.next != null) {
      fast = fast.next.next;
      slow = slow.next;
    }
    if (fast.next == null) {
      center = slow;
    } else {
      center = slow.next;
    }
    //reverse second half
    ListNode second = center.next;
    center.next = null;
    second = reverse(second);

    //merge
    ListNode first = A;
    ListNode temp;
    while (first != null && second != null) {
      temp = second;
      second = second.next;
      temp.next = first.next;
      first.next = temp;
      first = first.next.next;
    }
    return A;
  }

  private static ListNode reverse(ListNode center) {
    ListNode cur = center, prev = null, next;
    while (cur != null) {
      next = cur.next;
      cur.next = prev;
      prev = cur;
      cur = next;
    }
    return prev;
  }

  static class ListNode {
    public int val;
    public ListNode next;

    ListNode(int x) {
      val = x;
      next = null;
    }
  }


  public static class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
      val = x;
      left = null;
      right = null;
    }
  }

}
