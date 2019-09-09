import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {


  public static void main(String[] args) {
   new DP().coinChangeDP(new int[]{1,2,5}, 11);
  }

  public int[][] kClosest(int[][] points, int k) {
    PriorityQueue<Pair<Double, Integer>> pq = new PriorityQueue<>(new Comparator<Pair<Double, Integer>>() {
      @Override
      public int compare(Pair<Double, Integer> o1, Pair<Double, Integer> o2) {
        return o1.first.compareTo(o2.first);
      }
    });
    for (int i = 0; i < points.length; i++) {
      int[] point = points[i];
      double curDistance = Math.sqrt(point[0] * point[0] + point[1] * point[1]);
      if (pq.size() < k) {
        pq.add(new Pair<>(curDistance, i));
      } else {
        double maxDistance = pq.peek().first;
        if (curDistance < maxDistance) {
          pq.poll();
          pq.add(new Pair<>(curDistance, i));
        }
      }
    }
    int[][] res = new int[k][2];
    int idx = 0;
    while (!pq.isEmpty()) {
      Integer index = pq.poll().second;
      res[idx++] = new int[]{points[index][0], points[index][1]};
    }
    return res;
  }


  private int howManyAgentsToAdd(int noOfCurAgents, List<List<Integer>> callsTimes) {
    List<Pair<Integer, Boolean>> timeline = new ArrayList<>();
    for (List<Integer> call : callsTimes) {
      timeline.add(new Pair<>(call.get(0), true));
      timeline.add(new Pair<>(call.get(1), false));
    }
    timeline.sort((o1, o2) -> {
      int c = o1.first.compareTo(o2.first);
      if (c != 0) {
        return c;
      }
      if (o1.second) {
        return -1;
      }
      if (o2.second) {
        return 1;
      }
      return 0;
    });
    int ctr = 0;
    int maxEmployees = 0;
    for (Pair<Integer, Boolean> p : timeline) {
      if (p.second) {
        ctr++;
        maxEmployees = Math.max(maxEmployees, ctr);
      } else {
        ctr--;
      }
    }
    if (maxEmployees <= noOfCurAgents) {
      return 0;
    }
    return maxEmployees - noOfCurAgents;
  }


  private List<Integer> findMultiples(int x, int y, int z, int n) {
    List<Integer> res = new ArrayList<>();
    for (int i = 1; i <= n; i++) {
      if (((x != 0 && i % x == 0) || (y != 0 && i % y == 0))
          && (z != 0 && i % z != 0)) {
        res.add(i);
      }
    }
    return res;
  }


  public List<List<String>> accountsMerge(List<List<String>> accounts) {
    Map<String, HashSet<String>> res = new HashMap<>();
    int counter = 0;

    for (List<String> account : accounts) {

      HashSet<String> curEmails = new HashSet<>();
      String name = account.get(0);
      for (int idx = 1; idx < account.size(); idx++) {
        String email = account.get(idx);
        curEmails.add(email);
      }

      System.out.println("processing " + name);

      boolean addToPrevious;
      List<String> toRemove = new ArrayList<>();
      for (Map.Entry<String, HashSet<String>> entry : res.entrySet()) {
        String key = entry.getKey();
        String prevName = entry.getKey().split("#")[0];
        Set<String> prevEmails = entry.getValue();
        addToPrevious = false;
        if (prevName.equals(name)) {
          for (String prevEmail : prevEmails) {
            if (curEmails.contains(prevEmail)) {
              System.out.println("found common " + prevEmail);
              addToPrevious = true;
              break;
            }
          }
        }
        if (addToPrevious) {

          curEmails.addAll(prevEmails);
          toRemove.add(key);
        }
      }
      for (String r : toRemove) {
        res.remove(r);
      }
      res.put(name + "#" + ++counter, curEmails);
    }
    List<List<String>> ans = new ArrayList<>();
    for (Map.Entry<String, HashSet<String>> entry : res.entrySet()) {
      List<String> cur = new ArrayList<>();
      cur.add(entry.getKey().split("#")[0]);
      List<String> emails = new ArrayList<>(entry.getValue());
      Collections.sort(emails);
      cur.addAll(emails);
      ans.add(cur);
    }

    return ans;
  }

  public int largestUniqueNumber(int[] a) {
    int[] count = new int[1001];
    Arrays.fill(count, 0);
    for (int num : a) {
      count[num]++;
    }
    for (int i = 1000; i >= 0; i--) {
      if (count[i] == 1) {
        return i;
      }
    }
    return -1;
  }

  public int longestWPI(int[] h) {
    int n = h.length;
    Map<Integer, Integer> indices = new HashMap<>();
    int sum = 0;
    int res = 0;
    for (int i = 0; i < n; i++) {
      sum += h[i];
      if (sum > 0) {
        res = i + 1;
      } else {
        indices.putIfAbsent(sum, i);
        if (indices.containsKey(sum - 1)) {
          res = Math.max(res, i - indices.get(sum - 1) + 1);
        }
      }
    }
    return res;
  }

  public int[] distributeCandies(int candies, int num_people) {
    int[] people = new int[num_people];

    int c = 0;
    int left = candies;

    while (left > 0) {
      for (int i = 0; i < num_people && left > 0; i++) {
        people[i] += Math.min(c * num_people + (i + 1), left);
        left -= c * num_people + (i + 1);
      }
      c++;
    }
    return people;
  }

  public int leastInterval(char[] tasks, int n) {
    PriorityQueue<Integer> pq = new PriorityQueue<>(new Comparator<Integer>() {
      @Override
      public int compare(Integer o1, Integer o2) {
        return o2 - o1;
      }
    });
    Map<Character, Integer> map = new HashMap<>();
    for (char c : tasks) {
      map.put(c, map.getOrDefault(c, 0) + 1);
    }
    pq.addAll(map.values());
    int res = 0;
    while (!pq.isEmpty()) {
      int k = n;
      List<Integer> temp = new ArrayList<>();
      while (k >= 0 && !pq.isEmpty()) {
        Integer poll = pq.poll();
        poll--;
        temp.add(poll);
        k--;
        res++;
      }
      for (int t : temp) {
        if (t > 0) {
          pq.add(t);
        }
      }

      if (pq.isEmpty()) break;

      res += k;
    }
    return res;
  }

  public boolean carPooling(int[][] trips, int capacity) {
    List<int[]> times = new ArrayList<>();
    for (int[] trip : trips) {
      times.add(new int[]{trip[1], trip[0]});
      times.add(new int[]{trip[2], -trip[0]});
    }
    times.sort(new Comparator<int[]>() {

      @Override
      public int compare(int[] o1, int[] o2) {
        if (o1[0] != o2[0]) {
          return o1[0] - o2[0];
        }
        return o1[1] - o2[1];
      }
    });
    int c = 0;
    for (int[] trip : trips) {
      c += trip[1];
      if (c > capacity) {
        return false;
      }
    }
    return true;
  }

  public class NestedIterator implements Iterator<Integer> {

    Deque<NestedInteger> deque;

    public NestedIterator(List<NestedInteger> nestedList) {
      deque = new ArrayDeque<>();
      flattenList(nestedList);
    }

    @Override
    public Integer next() {
      return deque.pop().getInteger();
    }

    @Override
    public boolean hasNext() {
      while (!deque.isEmpty()) {
        if (deque.peek().isInteger()) return true;
        flattenList(deque.pop().getList());
      }
      return false;
    }

    void flattenList(List<NestedInteger> list) {
      for (int i = list.size() - 1; i >= 0; i--) {
        deque.push(list.get(i));
      }
    }
  }

  public int getImportance(List<Employee> employees, int id) {
    Map<Integer, List<Integer>> relation = new HashMap<>();
    Map<Integer, Integer> importance = new HashMap<>();
    for (Employee employee : employees) {
      relation.put(employee.id, employee.subordinates);
      importance.put(employee.id, employee.importance);
    }
    Queue<Integer> q = new LinkedList<>();
    q.add(id);
    int res = 0;
    while (!q.isEmpty()) {
      Integer polled = q.poll();
      res += importance.get(polled);
      q.addAll(relation.get(polled));
    }
    return res;
  }

  class Employee {
    // It's the unique id of each node;
    // unique id of this employee
    public int id;
    // the importance value of this employee
    public int importance;
    // the id of direct subordinates
    public List<Integer> subordinates;
  }

  public int depthSum(List<NestedInteger> nestedList) {
    return depthSum(nestedList, 1);
  }

  private int depthSum(List<NestedInteger> list, int multiplier) {
    int sum = 0;
    for (NestedInteger nestedInteger : list) {
      if (nestedInteger.isInteger()) {
        sum += (nestedInteger.getInteger() * multiplier);
      } else {
        sum += depthSum(nestedInteger.getList(), multiplier + 1);
      }
    }
    return sum;
  }

  public interface NestedInteger {
    // Constructor initializes an empty nested list.
//          pulic NestedInteger();

    // Constructor initializes a single integer.
//          public NestedIntegerdInteger(int value);

    // @return true if this NestedInteger holds a single integer, rather than a nested list.
    public boolean isInteger();

    // @return the single integer that this NestedInteger holds, if it holds a single integer
    // Return null if this NestedInteger holds a nested list
    public Integer getInteger();

    // Set this NestedInteger to hold a single integer.
    public void setInteger(int value);

    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
    public void add(NestedInteger ni);

    // @return the nested list that this NestedInteger holds, if it holds a nested list
    // Return null if this NestedInteger holds a single integer
    public List<NestedInteger> getList();
  }

  public int lastStoneWeight(int[] stones) {
    PriorityQueue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
      @Override
      public int compare(Integer o1, Integer o2) {
        return o2.compareTo(o1);
      }
    });
    for (int s : stones) {
      q.add(s);
    }
    while (!q.isEmpty()) {
      int a = q.poll();
      if (q.isEmpty()) {
        return a;
      }
      int b = q.poll();
      if (a - b > 0) {
        q.add(a - b);
      }
    }
    return 0;
  }

  public boolean isRobotBounded(String instructions) {
    int x = 0, y = 0, dir = 0;
    int count = 0;
    while (count < 3) {
      for (int i = 0; i < instructions.length(); i++) {
        switch (instructions.charAt(i)) {
          case 'G':
            if (dir == 0) y++;
            else if (dir == 1) x--;
            else if (dir == 2) y--;
            else if (dir == 3) x++;
            break;
          case 'L':
            if (dir == 3) dir = 0;
            else {
              dir++;
            }
            break;
          case 'R':
            if (dir == 0) dir = 3;
            else {
              dir--;
            }
            break;
        }
      }
      count++;
    }
    return (x == 0 && y == 0);
  }

  public boolean isBoomerang(int[][] points) {
    Arrays.sort(points, (o1, o2) -> {
      int c = Integer.compare(o1[0], o2[0]);
      if (c == 0) {
        return o1[1] - o2[1];
      }
      return c;
    });
    int numx = 1, numy = 1;
    if (points[1][0] != points[0][0]) {
      numx++;
    }
    if (points[2][0] != points[1][0]) {
      numx++;
    }
    if (points[1][1] != points[0][1]) {
      numy++;
    }
    if (points[2][1] != points[1][1]) {
      numy++;
    }
    if (numx < 2 || numy < 2) {
      return false;
    }

    int c = 1;
    if (points[1][0] != points[0][0] || points[1][1] != points[0][1]) {
      c++;
    }
    if (points[2][0] != points[1][0] || points[2][1] != points[1][1]) {
      c++;
    }
    if (c < 3) {
      return false;
    }

    if (points[1][0] - points[0][0] == points[2][0] - points[1][0] &&
        points[1][1] - points[0][1] == points[2][1] - points[1][1]) {
      return false;
    }
    return true;
  }

  public int[] numMovesStones(int a, int b, int c) {
    int[] distances = new int[]{a, b, c};
    Arrays.sort(distances);
    int x = distances[1] - distances[0] - 1, y = distances[2] - distances[1] - 1;
    int min = 0;
    int minD = Math.min(x, y), maxD = Math.max(x, y);
    if (minD == 0 && maxD == 0) {
      min = 0;
    } else if (minD == 0 || minD == 1) {
      min = 1;
    } else if (minD == 2) {
      min = 2;
    }
    int max = x + y;
    return new int[]{min, max};
  }


  public int generalizedGCD(int num, int[] arr) {
    int gcd = arr[0];
    for (int i = 1; i < arr.length; i++) {
      gcd = GCD(gcd, arr[i]);
    }
    return gcd;
  }

  private int GCD(int a, int b) {
    if (a < b) {
      return GCD(b, a);
    }
    if (b == 0) {
      return a;
    }
    return GCD(b, a % b);
  }


  public int solution(int[] A) {
    Arrays.sort(A);
    int i = 1;
    while (i <= Integer.MAX_VALUE) {
      if (Arrays.binarySearch(A, i) == -1) {
        return i;
      }
      if (i == Integer.MAX_VALUE) {
        break;
      }
      i++;
    }
    return -1;
  }


  public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
    List<Pair<Pair<Integer, Integer>, Integer>> distances = new ArrayList<>();
    for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
        int d = Math.abs(r0 - i) + Math.abs(c0 - j);
        distances.add(new Pair<>(new Pair<>(i, j), d));
      }
    }
    distances.sort(Comparator.comparingInt(o -> o.second));
    int[][] ans = new int[R * C][2];
    for (int i = 0; i < distances.size(); i++) {
      ans[i] = new int[2];
      ans[i][0] = distances.get(i).first.first;
      ans[i][1] = distances.get(i).first.second;
    }
    return ans;
  }

  public int addDigits(int num) {
    while (num >= 10) {
      int t = num;
      int sum = 0;
      while (t > 0) {
        sum += (t % 10);
        t /= 10;
      }
      num = sum;
    }
    return num;
  }

  public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
    return isRectangleOverlapUtil(rec1, rec2) && isRectangleOverlapUtil(rec2, rec1);
  }

  private boolean isRectangleOverlapUtil(int[] rec1, int[] rec2) {
    return (rec1[2] > rec2[0] && rec1[3] > rec2[1]);
  }

  public boolean divisorGame(int n) {
    int[] state = new int[n + 1];
    state[1] = 0;
    state[2] = 1;
    state[3] = 0;
    for (int i = 4; i <= n; i++) {
      state[i] = 0;
      for (int j = 1; j < i; j++) {
        if (i % j == 0 && state[i - j] == 0) {
          state[i] = 1;
          break;
        }
      }
    }
    return state[n] == 1;
  }

  public boolean isPowerOfThree(int n) {
    if (n <= 0) return false;
    if (n == 1 || n == 3) return true;
    int sum = 0;
    while (n > 0) {
      sum += n % 10;
      n /= 10;
    }
    return (sum % 9 == 0);
  }

  public boolean isPowerOfTwo(int n) {
    return (n & (n - 1)) == 0;
  }


  public int[] countBitsII(int num) {
    int[] dp = new int[num + 1];
    dp[0] = 0;
    if (num == 0) return dp;
    dp[1] = 1;
    int idx = 2, count = 2;
    while (idx <= num) {
      dp[idx] = dp[idx - count] + 1;
      idx++;
      if (idx == count * 2) {
        count *= 2;
      }
    }
    return dp;
  }

  public int[] countBits(int num) {
    int[] dp = new int[num + 1];
    dp[0] = 0;
    dp[1] = 1;
    int idx = 2, count = 1, previous = 1;
    while (idx <= num) {
      for (int i = idx, j = previous; i <= idx + count && i <= num; i++, j++) {
        dp[i] = dp[j];
      }
      for (int i = idx + count, j = previous; i <= idx + 2 * count && i <= num; i++, j++) {
        dp[i] = dp[j] + 1;
      }
      count *= 2;
      previous = idx;
      idx = idx + count;
    }
    return dp;
  }

  public int islandPerimeterII(int[][] grid) {
    int m = grid.length;
    if (m == 0) return 0;
    int n = grid[0].length;
    int[] xd = new int[]{0, -1, 0, 1};
    int[] yd = new int[]{-1, 0, 1, 0};
    int peri = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] == 1) {
          int newX, newY;
          for (int k = 0; k < 4; k++) {
            newX = i + xd[k];
            newY = j + yd[k];
            if (!(newX >= 0 && newY >= 0 && newX < m && newY < n) || grid[newX][newY] == 0) {
              peri++;
            }
          }
        }
      }
    }
    return peri;
  }


  public int islandPerimeter(int[][] grid) {
    int m = grid.length;
    if (m == 0) return 0;
    int n = grid[0].length;

    boolean[][] visited = new boolean[m][n];
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) visited[i][j] = false;

    Deque<Point> q = new LinkedList<Point>();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] == 1) {
          q.addLast(new Point(i, j));
          visited[i][j] = true;
          break;
        }
      }
      if (!q.isEmpty()) {
        break;
      }
    }
    int peri = 0;
    int[] xd = new int[]{0, -1, 0, 1};
    int[] yd = new int[]{-1, 0, 1, 0};
    while (!q.isEmpty()) {
      Point point = q.pollFirst();
      int newX, newY;
      for (int i = 0; i < 4; i++) {
        newX = point.x + xd[i];
        newY = point.y + yd[i];
        if (isValidPoint(m, n, newX, newY)) {
          if (grid[newX][newY] == 0) {
            peri++;
          } else if (!visited[newX][newY]) {
            visited[newX][newY] = true;
            q.addLast(new Point(newX, newY));
          }
        } else {
          peri++;
        }
      }
    }
    return peri;
  }

  private boolean isValidPoint(int m, int n, int newx, int newy) {
    return newx >= 0 && newy >= 0 && newx < m && newy < n;
  }


  public int maximumSwap(int num) {
    ArrayList<Integer> arr = new ArrayList<>();
    while (num > 0) {
      arr.add(num % 10);
      num /= 10;
    }
    if (arr.size() == 0) {
      return 0;
    }
    int maxNum, maxNumIdx, idx2 = -1;
    maxNum = arr.get(0);
    maxNumIdx = 0;
    for (int i = 1; i < arr.size(); i++) {
      if (arr.get(i) < maxNum) {
        idx2 = i;
      } else if (arr.get(i) > maxNum) {
        maxNum = arr.get(i);
        maxNumIdx = i;
      }
    }
    if (idx2 == -1) {
      return num;
    }

    Collections.swap(arr, idx2, maxNumIdx);
    int ans = 0;
    for (int i = 0; i < arr.size(); i++) {
      ans += arr.get(i) * Math.pow(10, i);
    }
    return ans;
  }

  public static int smallestRepunitDivByK(int k) {
    long n = 0;
    int i = 0;
    while (i < 1e6) {
      n = (n * 10 + 1);
      if (n % k == 0) {
        return (i + 1);
      }
      i++;
    }
    return -1;
  }

  public int maxScoreSightseeingPair(int[] A) {
    int n = A.length;
    int[] indices = new int[n];
    indices[n - 1] = -1;
    int ans = Integer.MIN_VALUE;
    for (int i = n - 2; i >= 0; i--) {
      int j = i + 1;
      indices[i] = j;
      int curVal = A[indices[i]] + i - indices[i];
      int proposedValue = (indices[j] != -1) ? A[indices[j]] + i - indices[j] : Integer.MIN_VALUE;
      while (proposedValue > curVal) {
        indices[i] = indices[j];
        j = indices[j];
        curVal = A[indices[i]] + i - indices[i];
        proposedValue = (indices[j] != -1) ? A[indices[j]] + i - indices[j] : Integer.MIN_VALUE;
      }
      ans = Math.max(ans, curVal + A[i]);
    }
    return ans;
  }

  public boolean canThreePartsEqualSum(int[] A) {
    int sum = 0;
    for (int aA : A) {
      sum += aA;
    }
    if (sum % 3 != 0) {
      return false;
    }
    int curSum = 0;
    int count = 0;
    for (int aA : A) {
      curSum += aA;
      if (curSum == sum / 3) {
        curSum = 0;
        count++;
      }
    }
    return (count >= 3);
  }

  public int trap(int[] height) {
    int n = height.length;
    Stack<Integer> st = new Stack<>();
    int idx = 0;
    int totalArea = 0;
    while (idx < n) {
      while (!st.empty() && height[idx] > height[st.peek()]) {
        int popped = st.pop();
        if (st.empty()) {
          break;
        }
        int minH = Math.min(height[idx], height[st.peek()]) - height[popped];
        int d = idx - st.peek() - 1;
        totalArea += minH * d;
      }
      st.push(idx);
      idx++;
    }
    return totalArea;
  }

  public boolean judgeSquareSum(int x) {
    if (x == 1) {
      return true;
    }
    long l = 1, r = (long) Math.sqrt(x);
    while (l <= r) {
      long cSum = l * l + r * r;
      if (cSum == x) {
        return true;
      } else if (cSum < x) {
        l++;
      } else {
        r--;
      }
    }
    return false;
  }

  public boolean isPerfectSquare(int x) {
    if (x == 1) {
      return true;
    }
    int l = 1, r = x / 2;
    while (l <= r) {
      int mid = r - (r - l) / 2;
      if (mid == (x / mid)) {
        return (x % mid == 0);
      } else if (mid < (x / mid)) {
        l = mid + 1;
      } else if (mid > (x / mid)) {
        r = mid - 1;
      }
    }
    return false;
  }

  public int mySqrt(int x) {
    if (x == 0 || x == 1) {
      return x;
    }

    int l = 1, r = x / 2;
    while (l <= r) {
      int mid = r - (r - l) / 2;
      if (mid == (x / mid)) {
        return mid;
      } else if (mid < (x / mid)) {
        if (((mid + 1) > (x / (mid + 1)))) {
          return mid;
        }
        l = mid;
      } else if (mid > (x / mid)) {
        r = mid;
      }
    }
    return 0;
  }

  public int[] singleNumber3(int[] nums) {
    int res = 0;
    for (int num1 : nums) {
      res ^= num1;
    }
    int b = 0;
    for (int i = 0; i < 32; i++) {
      if ((res & (1 << i)) != 0) {
        b = (1 << i);
        break;
      }
    }
    int res1 = res;
    for (int num : nums) {
      if ((num & b) != 0) {
        res1 ^= num;
      }
    }
    int[] ans = new int[2];
    ans[0] = res1;
    ans[1] = res ^ res1;
    return ans;
  }

  public int singleNumber(int[] nums) {
    int k = 3;
    int[] bitCount = new int[32];
    for (int n : nums) {
      for (int i = 0; i < 32; i++) {
        if ((n & (1 << i)) != 0) {
          bitCount[i]++;
        }
      }
    }
    int ans = 0;
    for (int i = 0; i < 32; i++) {
      bitCount[i] %= k;
      if (bitCount[i] != 0) {
        ans |= (1 << i);
      }
    }
    return ans;
  }

  public int numPairsDivisibleBy60(int[] time) {
    int count = 0;
    Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
    for (int i = 0; i < time.length; i++) {
      Pair<Integer, Integer> p = new Pair<>(time[i], i + 1);
      if (map.containsKey(p)) {
        count += map.get(p);
        continue;
      }
      int tCount = 0;
      for (int j = i + 1; j < time.length; j++) {
        if ((time[i] + time[j]) % 60 == 0) {
          tCount++;
          count++;
        }
      }
      map.put(p, tCount);
    }
    return count;
  }

  public static int bitwiseComplement(int N) {
    if (N == 0) return 1;
    int ans = 0;
    int i = 0;
    while (N > 0) {
      int i1 = 1 << i;
      if ((N & 1) == 0) {
        ans |= i1;
      }
      N >>= 1;
      i++;
    }
    return ans;
  }


  public int minDistance(int height, int width, int[] tree, int[] squirrel, int[][] nuts) {

    int[] distancesFromTree = new int[nuts.length];
    int toStart = -1, maxDiff = Integer.MIN_VALUE;

    for (int i = 0; i < nuts.length; i++) {
      int dFromTree = Math.abs(nuts[i][0] - tree[0]) + Math.abs(nuts[i][1] - tree[1]);
      distancesFromTree[i] = dFromTree;
      int distanceFromS = Math.abs(nuts[i][0] - squirrel[0]) + Math.abs(nuts[i][1] - squirrel[1]);
      if (dFromTree - distanceFromS > maxDiff) {
        maxDiff = dFromTree - distanceFromS;
        toStart = i;
      }
    }

    int ans = 0;
    ans += Math.abs(nuts[toStart][0] - squirrel[0]) + Math.abs(nuts[toStart][1] - squirrel[1]);

    for (int i = 0; i < distancesFromTree.length; i++) {
      if (i == toStart) {
        ans += distancesFromTree[i];
      } else {
        ans += (distancesFromTree[i] * 2);
      }
    }
    return ans;
  }


  public int distributeCandies(int[] candies) {
    Map<Integer, Integer> map = new HashMap<>();
    int total = candies.length;
    for (int c : candies) {
      map.put(c, map.getOrDefault(c, 0) + 1);
    }
    return Math.min(total / 2, map.size());
  }


  public int maximalRectangle2(char[][] matrix) {
    int m = matrix.length;
    if (m == 0) return 0;
    int n = matrix[0].length;
    int[] heights = new int[n];
    for (int i = 0; i < n; i++) {
      heights[i] = 0;
    }
    int maxArea = 0;
    for (char[] row : matrix) {
      for (int j = 0; j < n; j++) {
        if (row[j] == '0') {
          heights[j] = 0;
        } else {
          heights[j] = heights[j] + 1;
        }
      }
      maxArea = Math.max(maxArea, largestRectangleArea(heights));
    }
    return maxArea;
  }

  public int largestRectangleArea(int[] heights) {
    int n = heights.length;
    if (n == 0) return 0;

    int[] leftMinIdx = new int[n];
    leftMinIdx[0] = -1;
    int i = 1;
    while (i < n) {
      int j = i - 1;
      while (j >= 0 && heights[j] >= heights[i]) {
        j = leftMinIdx[j];
      }
      leftMinIdx[i] = j;
      i++;
    }

    int[] rightMinIdx = new int[n];
    rightMinIdx[n - 1] = n;
    i = n - 2;
    while (i >= 0) {
      int j = i + 1;
      while (j < n && heights[j] >= heights[i]) {
        j = rightMinIdx[j];
      }
      rightMinIdx[i] = j;
      i--;
    }
    i = 0;
    int maxArea = 0;
    while (i < n) {
      int l = leftMinIdx[i] + 1;
      int r = rightMinIdx[i] - 1;
      int curArea = heights[i] * (r - l + 1);
      maxArea = Math.max(maxArea, curArea);
      i++;
    }
    return maxArea;
  }

  public int chordCnt(int n) {
    long mod = 1000000007;
    if (n <= 2) {
      return n;
    }
    long[] dp = new long[n + 1];
    dp[0] = 1;
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; i++) {
      long temp = 0;
      for (int k = 0; k < i; k++) {
        temp = (int) ((temp + ((dp[k] * dp[i - k - 1]) % mod)) % mod);
      }
      dp[i] = temp;
    }
    return (int) dp[n];
  }

  public int maximalRectangle(char[][] matrix) {
    int m = matrix.length;
    if (m == 0) {
      return 0;
    }
    ArrayList<ArrayList<Integer>> A = new ArrayList<>();
    for (char[] aMatrix : matrix) {
      ArrayList<Integer> list = new ArrayList<>();
      for (int j = 0; j < aMatrix.length; j++) {
        if (aMatrix[j] == '0') {
          list.add(0);
        } else {
          list.add(1);
        }
      }
      A.add(list);
    }


    int n = A.get(0).size();
    int[][] sum = new int[m][n];
    sum[0][0] = A.get(0).get(0);
    for (int i = 1; i < n; i++) {
      sum[0][i] = sum[0][i - 1] + A.get(0).get(i);
    }
    for (int i = 1; i < m; i++) {
      sum[i][0] = sum[i - 1][0] + A.get(i).get(0);
    }
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + A.get(i).get(j);
      }
    }

    int ans = 0;

    for (int rsize = 1; rsize <= m; rsize++) {
      for (int csize = 1; csize <= n; csize++) {
        // submatrices of size (rsize,csize)
        for (int i = 0; i <= m - rsize; i++) {
          for (int j = 0; j <= n - csize; j++) {

            int row1 = i, row2 = i + rsize - 1, col1 = j, col2 = j + csize - 1;

            //submatrix of size (rsize,csize) starting at (row1,col1) and ending at (row2,col2)

            int s = sum[row2][col2];
            if (row1 == 0 && col1 == 0) {
              //do nothing
            }
            if (col1 != 0) {
              s -= sum[row2][col1 - 1];
            }
            if (row1 != 0) {
              s -= sum[row1 - 1][col2];
            }
            if (row1 != 0 && col1 != 0) {
              s += sum[row1 - 1][col1 - 1];
            }
            if (s == rsize * csize) {
              //it has all the ones
              ans = Math.max(ans, s);
            }

          }
        }
      }
    }
    return ans;
  }

  public int maximalRectangle(ArrayList<ArrayList<Integer>> A) {
    int m = A.size();
    if (m == 0) {
      return 0;
    }
    int n = A.get(0).size();
    int[][] sum = new int[m][n];
    sum[0][0] = A.get(0).get(0);
    for (int i = 1; i < n; i++) {
      sum[0][i] = sum[0][i - 1] + A.get(0).get(i);
    }
    for (int i = 1; i < m; i++) {
      sum[i][0] = sum[i - 1][0] + A.get(i).get(0);
    }
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + A.get(i).get(j);
      }
    }

    int ans = 0;

    for (int rsize = 1; rsize <= m; rsize++) {
      for (int csize = 1; csize <= n; csize++) {
        // submatrices of size (rsize,csize)
        for (int i = 0; i <= m - rsize; i++) {
          for (int j = 0; j <= n - csize; j++) {

            int row1 = i, row2 = i + rsize - 1, col1 = j, col2 = j + csize - 1;

            //submatrix of size (rsize,csize) starting at (row1,col1) and ending at (row2,col2)

            int s = sum[row2][col2];
            if (row1 == 0 && col1 == 0) {
              //do nothing
            }
            if (col1 != 0) {
              s -= sum[row2][col1 - 1];
            }
            if (row1 != 0) {
              s -= sum[row1 - 1][col2];
            }
            if (row1 != 0 && col1 != 0) {
              s += sum[row1 - 1][col1 - 1];
            }
            if (s == rsize * csize) {
              //it has all the ones
              ans = Math.max(ans, s);
            }

          }
        }
      }
    }
    return ans;
  }

  public static int solve(ArrayList<ArrayList<Integer>> A) {
    int m = A.size();
    if (m == 0) {
      return 0;
    }
    int n = A.get(0).size();
    int[][] sum = new int[m][n];
    sum[0][0] = A.get(0).get(0);
    for (int i = 1; i < n; i++) {
      sum[0][i] = sum[0][i - 1] + A.get(0).get(i);
    }
    for (int i = 1; i < m; i++) {
      sum[i][0] = sum[i - 1][0] + A.get(i).get(0);
    }
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + A.get(i).get(j);
      }
    }

    int ans = 0;

    for (int rsize = 1; rsize <= m; rsize++) {
      for (int csize = 1; csize <= n; csize++) {
        // submatrices of size (rsize,csize)
        for (int i = 0; i <= m - rsize; i++) {
          for (int j = 0; j <= n - csize; j++) {

            int row1 = i, row2 = i + rsize - 1, col1 = j, col2 = j + csize - 1;

            //submatrix of size (rsize,csize) starting at (row1,col1) and ending at (row2,col2)

            int s = sum[row2][col2];
            if (row1 == 0 && col1 == 0) {
              //do nothing
            }
            if (col1 != 0) {
              s -= sum[row2][col1 - 1];
            }
            if (row1 != 0) {
              s -= sum[row1 - 1][col2];
            }
            if (row1 != 0 && col1 != 0) {
              s += sum[row1 - 1][col1 - 1];
            }
            if (s == 0) {
              ans++;
            }
          }
        }
      }
    }
    return ans;
  }


  public int hammingWeight2(int n) {
    int ans = 0;
    while (n != 0) {
      ans++;
      n &= (n - 1);
    }
    return ans;
  }

  public int reverseBits(int n) {
    int c = 31, num = 1 << c;
    int ans = 0;
    while (c >= 0) {
      if ((num & n) != 0) {
        ans |= (1 << (31 - c));
      }
      c++;
      num = 1 << c;
    }
    return ans;
  }

  public int hammingWeight(int n) {
    int c = 0, num = 1;
    int distance = 0;
    while (c <= 31) {
      if ((num & n) != 0) {
        distance++;
      }
      c++;
      num = 1 << c;
    }
    return distance;
  }

  public int hammingDistance(int x, int y) {
    int c = 0, num = 1;
    int distance = 0;
    while (num <= x && num <= y && num > 0) {
      if ((num & x) != (num & y)) {
        distance++;
      }
      c++;
      num = 1 << c;
    }
    while (num <= x && num > 0) {
      if ((num & x) != 0) {
        distance++;
      }
      c++;
      num = 1 << c;
    }
    while (num <= y && num > 0) {
      if ((num & y) != 0) {
        distance++;
      }
      c++;
      num = 1 << c;
    }
    return distance;
  }

  public void moveZeroes(int[] nums) {
    int r = 0, l = 0;
    while (r < nums.length) {
      if (nums[r] != 0) {
        while (l <= r && nums[l] != 0) {
          l++;
        }
        if (l < r) {
          int temp = nums[r];
          nums[r] = nums[l];
          nums[l] = temp;
        }
      }
      r++;
    }
  }

  public int findPairs2(int[] nums, int k) {
    HashMap<Integer, Integer> pairs = new HashMap<>();
    int ans = 0;
    for (int num : nums) {
      pairs.put(num, pairs.getOrDefault(num, 0) + 1);
    }
    for (Map.Entry<Integer, Integer> entry : pairs.entrySet()) {
      if (k == 0) {
        if (entry.getValue() > 1) {
          ans++;
        }
      }
      if (pairs.containsKey(entry.getKey() + k)) {
        ans++;
      }
    }

    return ans;
  }

  public int findPairs(int[] nums, int k) {
    HashSet<Integer> numbers = new HashSet<>();
    HashSet<Pair> pairs = new HashSet<>();
    int ans = 0;
    for (int num : nums) {
      if (numbers.contains(num + k) && !pairs.contains(new Pair<>(num, num + k))) {
        ans++;
        pairs.add(new Pair<>(num, num + k));
      }
      if (numbers.contains(num - k) && !pairs.contains(new Pair<>(num, num - k))) {
        ans++;
        pairs.add(new Pair<>(num, num - k));
      }
      numbers.add(num);
    }
    return ans;
  }

  public int numRabbits(int[] answers) {
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int i : answers) {
      map.put(i, map.getOrDefault(i, 0) + 1);
    }
    int ans = 0;
    for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
      int total = entry.getValue();
      int colorValue = entry.getKey() + 1;
      ans += colorValue * (total / colorValue);
      if (total % colorValue != 0) {
        ans += colorValue;
      }
    }

    return ans;
  }


  public int findMaxConsecutiveOnes(int[] nums) {
    int maxCount = 0;
    for (int i = 0; i < nums.length; i++) {
      if (nums[i] == 0) continue;
      int curCount = 0;
      while (i < nums.length && nums[i] == 1) {
        curCount++;
        i++;
      }
      maxCount = Math.max(maxCount, curCount);
    }
    return maxCount;
  }

  public int firstUniqChar(String s) {
    Map<Character, Integer> m = new HashMap<>();
    for (int i = 0; i < s.length(); i++) {
      if (!m.containsKey(s.charAt(i))) {
        m.put(s.charAt(i), 0);
      }
      m.put(s.charAt(i), m.get(s.charAt(i)) + 1);
    }
    for (int i = 0; i < s.length(); i++) {
      if (m.get(s.charAt(i)) == 1) {
        return i;
      }
    }
    return -1;
  }

  public int[] getModifiedArray(int length, int[][] updates) {
    int[] ans = new int[length];
    for (int i = 0; i < length; i++) {
      ans[i] = 0;
    }
    for (int[] update : updates) {
      ans[update[0]] += update[2];
      if (update[1] < ans.length) {
        ans[update[1]] -= update[2];
      }
    }
    int cur = 0;
    for (int i = 0; i < length; i++) {
      cur += ans[i];
      ans[i] = cur;
    }
    return ans;
  }


  public int maxCount(int m, int n, int[][] ops) {
    int[] row = new int[m];
    for (int i = 0; i < m; i++) row[i] = 0;
    int[] col = new int[n];
    for (int i = 0; i < n; i++) col[i] = 0;

    for (int[] op : ops) {
      if (op[0] > 0) {
        row[op[0] - 1] += 1;
      }
      if (op[1] > 0) {
        col[op[1] - 1] += 1;
      }

      if (op[0] < op[1]) {
        if (op[0] > 0) {
          col[op[0] - 1] += -1;
        }
      } else {
        if (op[1] > 0) {
          row[op[1] - 1] += -1;
        }
      }
    }
    for (int i = m - 2; i >= 0; i--) {
      row[i] += row[i + 1];
    }

    for (int i = n - 2; i >= 0; i--) {
      col[i] += col[i + 1];
    }
    int max = -1;
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        int cur = row[i] + col[j];
        if (cur >= max) {
          max = cur;
          if (!map.containsKey(max)) {
            map.put(max, 0);
          }
          map.put(max, map.get(max) + 1);
        }
      }
    }
    return map.get(max);
  }

  public int minAreaRect2(int[][] points) {
    int n = points.length;
    HashSet<Point> pointHashSet = new HashSet<>();
    for (int[] point : points) {
      Point p = new Point(point[0], point[1]);
      pointHashSet.add(p);
    }
    int minArea = Integer.MAX_VALUE;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        if (points[i][0] == points[j][0] || points[i][1] == points[j][1]) {
          continue;
        }
        if (pointHashSet.contains(new Point(points[i][0], points[j][1])) && pointHashSet.contains(new Point(points[j][0], points[i][1]))) {
          int curBreadth = Math.abs(points[i][0] - points[j][0]);
          int curLength = Math.abs(points[i][1] - points[j][1]);
          int curArea = curBreadth * curLength;
          minArea = Math.min(minArea, curArea);
        }
      }
    }
    return (minArea != Integer.MAX_VALUE) ? minArea : 0;
  }

  class Point {
    int x, y;

    public Point(int x, int y) {
      this.x = x;
      this.y = y;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      Point point = (Point) o;
      return x == point.x &&
          y == point.y;
    }

    @Override
    public int hashCode() {
      return Objects.hash(x, y);
    }
  }

  public int minAreaRect(int[][] points) {
    HashMap<Integer, ArrayList<Integer>> m = new HashMap<>();
    int n = points.length;
    for (int[] point : points) {
      if (!m.containsKey(point[0])) {
        m.put(point[0], new ArrayList<>());
      }
      m.get(point[0]).add(point[1]);
    }
    int minArea = Integer.MAX_VALUE;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        if (points[i][0] == points[j][0] || points[i][1] == points[j][1]) {
          continue;
        }
        if (m.get(points[i][0]).contains(points[j][1]) && m.get(points[j][0]).contains(points[i][1])) {
          int curBreadth = Math.abs(points[i][0] - points[j][0]);
          int curLength = Math.abs(points[i][1] - points[j][1]);
          int curArea = curBreadth * curLength;
          minArea = Math.min(minArea, curArea);
        }
      }
    }
    return minArea;
  }

  public static int[][] candyCrush(int[][] board) {
    int m = board.length, n = board[0].length;
    boolean isStable = false;
    while (!isStable) {
      isStable = true;

      for (int i = 0; i < m; i++) {
        int j = 0;
        while (j < n) {
          int curVal = Math.abs(board[i][j]);
          int curStart = j;
          j++;
          while (j < n && Math.abs(board[i][j]) != 0 && Math.abs(board[i][j]) == curVal) {
            j++;
          }
          if (j - curStart >= 3) {
            // flag
            for (int k = curStart; k < j; k++) {
              board[i][k] = -1 * Math.abs(board[i][k]);
            }
          }
        }
      }


      for (int j = 0; j < n; j++) {
        int i = 0;
        while (i < m) {
          int curVal = Math.abs(board[i][j]);
          int curStart = i;
          i++;
          while (i < m && Math.abs(board[i][j]) != 0 && Math.abs(board[i][j]) == curVal) {
            i++;
          }
          if (i - curStart >= 3) {
            // flag
            for (int k = curStart; k < i; k++) {
              board[k][j] = -1 * Math.abs(board[k][j]);
            }
          }
        }
      }
      System.out.println("flagging done");

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          if (board[i][j] < 0) {
            isStable = false;
            board[i][j] = 0;
          }
        }
      }

      System.out.println("crushing done");

      for (int j = 0; j < n; j++) {
        int pointer = m;
        int i = m - 1;
        while (i >= 0) {
          if (board[i][j] != 0) {
            pointer--;
            int t = board[pointer][j];
            board[pointer][j] = board[i][j];
            board[i][j] = t;
          }
          i--;
        }
      }
      System.out.println("gravity done");
    }
    print(board);
    return board;
  }

  private static void print(int[][] a) {
    for (int[] anA : a) {
      for (Integer i : anA) {
        System.out.print(i + " ");
      }
      System.out.println();
    }
  }

  public int maxIncreaseKeepingSkyline(int[][] grid) {
    int n = grid.length;
    if (n == 0) {
      return 0;
    }
    int[] rowMax = new int[n];
    for (int i = 0; i < n; i++) {
      int mx = grid[i][0];
      for (int j = 1; j < n; j++) {
        mx = Math.max(mx, grid[i][j]);
      }
      rowMax[i] = mx;
    }
    int[] colMax = new int[n];
    for (int i = 0; i < n; i++) {
      int mx = grid[0][i];
      for (int j = 1; j < n; j++) {
        mx = Math.max(mx, grid[j][i]);
      }
      colMax[i] = mx;
    }
    int total = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] < rowMax[i] && grid[i][j] < colMax[j]) {
          total += (Math.min(rowMax[i], colMax[j]) - grid[i][j] + 1);
        }
      }
    }
    return total;
  }


  private int findRoot(int i, int[] root) {
    while (root[i] != i) {
      i = root[i];
    }
    return i;
  }


  public static int solve(final List<Integer> A, final List<Integer> B, final List<Integer> C) {
    int[][] dp = new int[B.size() + 1][1001];
    for (int j = 0; j < 1001; j++) {
      dp[0][j] = 9999;
    }
    for (int i = 0; i < B.size() + 1; i++) {
      dp[i][0] = 0;
    }
    for (int i = 1; i < B.size() + 1; i++) {
      for (int j = 1; j < 1001; j++) {
        dp[i][j] = dp[i - 1][j];
        Integer curCapacity = B.get(i - 1);
//        if (curCapacity <= j) {
//          dp[i][j] = Math.min(dp[i][j], dp[i - 1][j - curCapacity] + C.get(i - 1));
//        }
        if (j - curCapacity >= 0) {
          dp[i][j] = Math.min(dp[i][j], dp[i][j - curCapacity] + C.get(i - 1));
        }
      }
    }
//    for (int i = 0; i < B.size() + 1; i++) {
//      for (int j = 0; j < 15; j++) {
//        System.out.print(dp[i][j] + " ");
//      }
//      System.out.println();
//    }
    int ans = 0;
    for (Integer i : A) {
      ans += dp[B.size()][i];
    }
    return ans;
  }

  Map<TreeNode, IntegerPair> map = new HashMap<>();

  public int rob(TreeNode root) {
    map.put(null, new IntegerPair(0, 0));
    if (map.containsKey(root)) {
      return Math.max(map.get(root).first, map.get(root).second);
    }

    rob(root.left);
    rob(root.right);

    int lmax = 0, rmax = 0;
    IntegerPair leftPair = map.get(root.left);
    lmax = Math.max(leftPair.first, leftPair.second);

    IntegerPair rightPair = map.get(root.right);
    rmax = Math.max(rightPair.first, rightPair.second);

    map.put(root, new IntegerPair(lmax + rmax, root.val + leftPair.first + rightPair.first));

    return Math.max(map.get(root).first, map.get(root).second);
  }


  public static int rob2(int[] nums) {
    if (nums.length == 0) return 0;
    int solution = 0;
    int prev2 = 0, prev1 = 0;
    for (int i = 1; i < nums.length; i++) {
      int cur = Math.max(prev2 + nums[i], prev1);
      prev2 = prev1;
      prev1 = cur;
    }
    solution = prev1;
    prev2 = 0;
    prev1 = 0;
    for (int i = 0; i < nums.length - 1; i++) {
      int cur = Math.max(prev2 + nums[i], prev1);
      prev2 = prev1;
      prev1 = cur;
    }
    return Math.max(solution, prev1);
  }

  int[][] dp;

  public int rob(int[] nums) {
    dp = new int[nums.length][2];
    for (int i = 0; i < nums.length; i++) {
      for (int j = 0; j < 2; j++) {
        dp[i][j] = -1;
      }
    }
    return calc(nums, 0, 0);
  }

  private int calc(int[] nums, int idx, int prev) {
    if (dp[idx][prev] != -1) {
      return dp[idx][prev];
    }
    if (idx == nums.length) {
      return 0;
    }
    if (prev == 1) {
      dp[idx][prev] = calc(nums, idx + 1, 0);
    }
    if (prev == 0) {
      dp[idx][prev] = Math.max(calc(nums, idx + 1, 0), calc(nums, idx + 1, 1) + nums[idx]);
    }
    return dp[idx][prev];
  }

  public static int getMaxRepetitions(String s1, int n1, String s2, int n2) {
    int idx = 0, count = 0, interCount = 0;
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < s1.length(); j++) {
        if (s1.charAt(j) == s2.charAt(idx)) {
          idx++;
        }
        if (idx == s2.length()) {
          if (interCount == n2 - 1) {
            count++;
            interCount = 0;
          } else {
            interCount++;
          }
          idx = 0;
        }
      }
    }
    if (idx == s2.length()) {
      if (interCount == n2 - 1) {
        count++;
      }
    }
    return count;
  }

  private static int checkIfCanObtainS2FromS1(String s1, String s2) {
    int i = 0, j = 0;
    int count = 0;
    while (i < s1.length()) {
      if (j == s2.length()) {
        j = 0;
        count++;
      }
      if (s1.charAt(i) == s2.charAt(j)) {
        i++;
        j++;
      } else {
        i++;
      }
    }
    if (j == s2.length()) {
      count++;
    }
    return count;
  }

  public static int divide(int A, int B) {
    long al = A, bl = B;
    boolean negate = false;
    if (al < 0 || bl < 0) {
      negate = true;
    }
    if (al < 0 && bl < 0) {
      negate = false;
    }
    al = Math.abs(al);
    bl = Math.abs(bl);
    long count = 1;
    while (bl <= al) {
      bl += B;
      count++;
    }
    if (count > Integer.MAX_VALUE) {
      return Integer.MAX_VALUE;
    } else {
      return (int) (count - 1) * (negate ? -1 : 1);
    }
  }

  public static ArrayList<Integer> sieve(int A) {
    int[] primes = new int[A + 1];
    for (int i = 0; i < A + 1; i++) {
      primes[i] = 1;
    }
    primes[0] = 0;
    primes[1] = 0;
    for (int i = 2; i <= Math.sqrt(A + 1); i++) {
      if (primes[i] == 1) {
        int j = i + i;
        while (j < A + 1) {
          primes[j] = 0;
          j += i;
        }
      }
    }
    ArrayList<Integer> ans = new ArrayList<>();
    for (int i = 0; i < A + 1; i++) {
      if (primes[i] == 1) {
        ans.add(i);
      }
    }
    return ans;
  }

  public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
    Stack<UndirectedGraphNode> st = new Stack<>();
    st.push(node);
    HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<>();
    map.put(node, new UndirectedGraphNode(node.label));
    while (!st.isEmpty()) {
      UndirectedGraphNode t = st.pop();
      for (UndirectedGraphNode n : t.neighbors) {
        UndirectedGraphNode neighbourClone;
        if (!map.containsKey(n)) {
          neighbourClone = new UndirectedGraphNode(n.label);
          map.put(n, neighbourClone);
          st.push(n);
        } else {
          neighbourClone = map.get(n);
        }
        map.get(t).neighbors.add(neighbourClone);
      }
    }
    return map.get(node);
  }

  class UndirectedGraphNode {
    int label;
    List<UndirectedGraphNode> neighbors;

    UndirectedGraphNode(int x) {
      label = x;
      neighbors = new ArrayList<>();
    }
  }

  ;

  public static int solveNew(int A, int B, int C, int D) {
    int x = D - 1;
    if (A % B == 0 || A % C == 0) {
      return solveNew(B, C, D);
    }
    if (B % A == 0 || B % C == 0) {
      return solveNew(A, C, D);
    }
    if (C % A == 0 || C % B == 0) {
      return solveNew(A, B, D);
    }
    int p1 = x / A, p2 = x / B, p3 = x / C;
    int p4 = x / lcm(A, B), p5 = x / lcm(B, C), p6 = x / lcm(A, C), p7 = x / lcm(A, lcm(B, C));
    return p1 + p2 + p3 - p4 - p5 - p6 + p7;
  }

  private static int solveNew(int A, int B, int D) {
    int x = D - 1;

    if (A % B == 0) {
      return x / B;
    }
    if (B % A == 0) {
      return x / A;
    }

    int p1 = x / A, p2 = x / B, p3 = x / (A * B);
    return p1 + p2 - p3;
  }

  static int gcd(int a, int b) {
    // base case
    if (a == b)
      return a;

    // first is greater
    if (a > b)
      return gcd(a - b, b);
    return gcd(a, b - a);
  }

  // Function to return LCM of two numbers
  static int lcm(int a, int b) {
    return (a * b) / gcd(a, b);
  }

  static long stockmax(int[] prices) {
    int[] maxOnRight = new int[prices.length];
    int maximumInRight = -1;
    long profit = 0;
    for (int i = prices.length - 1; i >= 0; i--) {
      maxOnRight[i] = maximumInRight;
      if (maximumInRight <= prices[i]) {
        maximumInRight = prices[i];
      }
      //profit += (maximumInRight - prices[i]);
    }
    // return profit;
    int numShares = 0;
    for (int i = 0; i < prices.length; i++) {
      if (prices[i] < maxOnRight[i]) {
        profit -= prices[i];
        numShares++;
      } else if (prices[i] > maxOnRight[i]) {
        profit += (numShares * prices[i]);
        numShares = 0;
      }
    }
    return profit;
  }

  static long stockmax2(int[] prices) {
    int maximumInRight = -1;
    long profit = 0;
    for (int i = prices.length - 1; i >= 0; i--) {
      if (maximumInRight <= prices[i]) {
        maximumInRight = prices[i];
      }
      profit += (maximumInRight - prices[i]);
    }
    return profit;
  }

  public static int seats(String A) {
    int MOD = 10000003;

    int totalVal = 0;
    for (int i = 0; i < A.length(); i++) {
      if (A.charAt(i) == 'x') {
        totalVal++;
      }
    }
    int median = (totalVal + 1) / 2;

    int count = 0;
    int medianIdx = -1;
    for (int i = 0; i < A.length(); i++) {
      if (A.charAt(i) == 'x') {
        count++;
      }
      if (count == median) {
        medianIdx = i;
        break;
      }
    }

    int totalCost = 0;
    int totalTillNow = 0;
    for (int i = 0; i < medianIdx; i++) {
      if (A.charAt(i) == 'x') {
        totalTillNow++;
      } else {
        totalCost = (totalCost % MOD + totalTillNow % MOD) % MOD;
      }
    }
    totalTillNow = 0;
    for (int i = A.length() - 1; i > medianIdx; i--) {
      if (A.charAt(i) == 'x') {
        totalTillNow++;
      } else {
        totalCost = (totalCost % MOD + totalTillNow % MOD) % MOD;
      }
    }
    return totalCost % MOD;
  }


  public static int fibsum(int N) {
    ArrayList<Integer> arr = new ArrayList<>();
    arr.add(1);
    int next = 1;
    while (next <= N) {
      arr.add(next);
      next = arr.get(arr.size() - 1) + arr.get(arr.size() - 2);
    }
    int count = 0;
    int idx = arr.size() - 1;
    while (N > 0) {
      int cur = arr.get(idx);
      while (cur <= N) {
        N -= cur;
        count++;
      }
      idx--;
    }
    return count;
  }

  public int solveUsingUnionFind(int A, ArrayList<ArrayList<Integer>> B) {
    PriorityQueue<EdgeInfo> edges = new PriorityQueue<>(Comparator.comparing(o -> o.weight));
    for (ArrayList<Integer> aB : B) {
      for (int j = 0; j < 3; j++) {
        edges.add(new EdgeInfo(aB.get(0), aB.get(1), aB.get(2)));
      }
    }
    UnionFind u = new UnionFind(A + 1);
    int numEdges = 1;
    int totalWeight = 0;
    while (numEdges <= A - 1) {
      EdgeInfo edge = edges.poll();
      if (!u.checkIfConnected(edge.v1, edge.v2)) {
        numEdges++;
        totalWeight += edge.weight;
        u.union(edge.v1, edge.v2);
      }
    }
    return totalWeight;
  }

  public int solveUsingAdjacencyList(int A, ArrayList<ArrayList<Integer>> B) {
    PriorityQueue<EdgeInfo> edges = new PriorityQueue<>(Comparator.comparing(o -> o.weight));
    for (ArrayList<Integer> aB : B) {
      for (int j = 0; j < 3; j++) {
        edges.add(new EdgeInfo(aB.get(0), aB.get(1), aB.get(2)));
      }
    }
    ArrayList<ArrayList<Integer>> graphList = new ArrayList<>(A + 1);
    for (int i = 0; i < A + 1; i++) {
      graphList.add(new ArrayList<>());
    }
    int numEdges = 1;
    int totalWeight = 0;
    while (numEdges <= A - 1) {
      EdgeInfo edge = edges.poll();
      if (!checkIfPathExists(edge.v1, edge.v2, graphList)) {
        numEdges++;
        totalWeight += edge.weight;
      }
      ArrayList<Integer> v2List = graphList.get(edge.v2);
      if (!v2List.contains(edge.v1)) {
        v2List.add(edge.v1);
      }
      ArrayList<Integer> v1List = graphList.get(edge.v1);
      if (!v1List.contains(edge.v2)) {
        v1List.add(edge.v2);
      }
    }
    return totalWeight;
  }

  private boolean checkIfPathExists(int startVertex, int endVertex, ArrayList<ArrayList<Integer>> adjacencyList) {
    ArrayList<Integer> visited = new ArrayList<>();
    Stack<Integer> currentGraph = new Stack<>();
    currentGraph.push(startVertex);
    visited.add(startVertex);
    while (!currentGraph.empty()) {
      int currentNode = currentGraph.pop();
      ArrayList<Integer> neighbours = adjacencyList.get(currentNode);
      for (Integer i : neighbours) {
        if (i == endVertex) {
          return true;
        }
        if (!visited.contains(i)) {
          visited.add(i);
          currentGraph.push(i);
        }
      }
    }
    return false;
  }

  class EdgeInfo {
    Integer weight, v1, v2;

    public EdgeInfo(Integer v1, Integer v2, Integer weight) {
      this.v1 = v1;
      this.v2 = v2;
      this.weight = weight;
    }
  }

  public static int solve(List<Integer> A) {
    Map<Integer, TreeNodeWithChildren> m = new HashMap<>();
    TreeNodeWithChildren root = null;
    for (int i = 0; i < A.size(); i++) {
      TreeNodeWithChildren curNode = new TreeNodeWithChildren(new ArrayList<>());
      if (!m.containsKey(i)) {
        m.put(i, curNode);
      } else {
        curNode = m.get(i);
      }
      Integer parentIdx = A.get(i);
      if (parentIdx == -1) {
        root = curNode;
      } else {
        TreeNodeWithChildren parent;
        if (!m.containsKey(parentIdx)) {
          parent = new TreeNodeWithChildren(new ArrayList<>());
          m.put(parentIdx, parent);
        } else {
          parent = m.get(parentIdx);
        }
        parent.addChild(curNode);
      }
    }
    return calcLargestPath(root);
  }

  static int calcLargestPath(TreeNodeWithChildren root) {
    int maxPath = 1;
    Stack<TreeNodeWithChildren> st = new Stack<>();
    st.push(root);
    while (!st.empty()) {
      TreeNodeWithChildren curNode = st.peek();
      boolean unvisitedChildLeft = false;
      for (int i = 0; i < curNode.children.size(); i++) {
        if (!curNode.children.get(i).visited) {
          unvisitedChildLeft = true;
          TreeNodeWithChildren newNode = curNode.children.get(i);
          newNode.visited = true;
          st.push(newNode);
        }
      }
      if (!unvisitedChildLeft) {
        st.pop();
        if (curNode.children.size() == 0) {
          curNode.depth = 1;
          continue;
        }
        ArrayList<Integer> depths = new ArrayList<>();
        for (int i = 0; i < curNode.children.size(); i++) {
          depths.add(curNode.children.get(i).depth);
        }
        depths.sort(Comparator.naturalOrder());
        int curMaxPath = depths.get(depths.size() - 1) + 1;
        if (depths.size() > 1) {
          curMaxPath += depths.get(depths.size() - 2);
        }
        if (curMaxPath > maxPath) {
          maxPath = curMaxPath;
        }
        curNode.depth = depths.get(depths.size() - 1) + 1;
      }

    }
    return maxPath - 1;
  }

  static class GraphNode {
    ArrayList<TreeNodeWithChildren> children;
    int value = 1;
    boolean visited = false;

    public GraphNode(ArrayList<TreeNodeWithChildren> children) {
      this.children = children;
    }

    public void addChild(TreeNodeWithChildren t) {
      children.add(t);
    }
  }

  static class TreeNodeWithChildren {
    ArrayList<TreeNodeWithChildren> children;
    int depth = 1;
    boolean visited = false;

    public TreeNodeWithChildren(ArrayList<TreeNodeWithChildren> children) {
      this.children = children;
    }

    public void addChild(TreeNodeWithChildren t) {
      children.add(t);
    }
  }

  public static ArrayList<Integer> solve(int A, int B, int C, int D) {
    ArrayList<Integer> ans = new ArrayList<>();
    Map<Integer, Integer> map = new HashMap<>();
    map.put(A, 1);
    map.put(B, 1);
    map.put(C, 1);
    Queue<Integer> q = new PriorityQueue<>();
    q.add(A);
    if (!map.containsKey(B)) {
      q.add(B);
    }
    if (!map.containsKey(C)) {
      q.add(C);
    }
    while (ans.size() < D) {
      int t = q.poll();
      ans.add(t);
      int m1 = t * C;
      int m2 = t * B;
      int m3 = t * A;
      if (!map.containsKey(m1)) {
        map.put(m1, 1);
        q.add(m1);
      }
      if (!map.containsKey(m2)) {
        map.put(m2, 1);
        q.add(m2);
      }
      if (!map.containsKey(m3)) {
        map.put(m3, 1);
        q.add(m3);
      }
    }
    for (Integer i : ans) {
      System.out.print(i + " ");
    }
    return new ArrayList<>(ans);
  }


  public static ArrayList<Integer> dNums(List<Integer> A, int B) {
    ArrayList<Integer> ans = new ArrayList<>();
    if (A.size() < B || B == 0) {
      return ans;
    }
    HashMap<Integer, Integer> counts = new HashMap<>();
    for (int i = 0; i < B; i++) {
      if (!counts.containsKey(A.get(i))) {
        counts.put(A.get(i), 1);
      } else {
        counts.put(A.get(i), counts.get(A.get(i)) + 1);
      }
    }
    ans.add(counts.size());
    for (int i = B; i < A.size(); i++) {
      int lostIndex = i - B;
      Integer lostNumber = A.get(lostIndex);
      if (counts.get(lostNumber) > 1) {
        counts.put(lostNumber, counts.get(lostNumber) - 1);
      } else {
        counts.remove(lostNumber);
      }
      Integer curNumber = A.get(i);
      if (!counts.containsKey(curNumber)) {
        counts.put(curNumber, 1);
      } else {
        counts.put(curNumber, counts.get(curNumber) + 1);
      }
      ans.add(counts.size());
    }
    for (Integer i : ans) {
      System.out.print(i);
    }
    return ans;
  }

  public ListNode mergeKLists(ArrayList<ListNode> a) {
    PriorityQueue<ListNode> heap = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
    heap.addAll(a);
    ListNode resultNode = null, resultTail = null;
    while (!heap.isEmpty()) {
      ListNode temp = heap.remove();
      if (resultNode == null) {
        resultNode = temp;
        resultTail = temp;
      } else {
        resultTail.next = temp;
        resultTail = temp;
      }
      if (temp.next != null) {
        heap.add(temp.next);
      }
    }
    return resultNode;
  }

  private static void scan() {
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    int testCases;
    try {
      testCases = Integer.parseInt(br.readLine());
      String line;

      for (int i = 0; i < testCases; i++) {
        line = br.readLine();
        ArrayList<Integer> array = flip(line);
        if (array.size() == 0) {
          System.out.println();
        } else {
          System.out.print(String.valueOf(array.get(0)));
          System.out.print(" ");
          System.out.print(String.valueOf(array.get(1)));
          if (i != testCases - 1) {
            System.out.println();
          }
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

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
    return new ArrayList<>(Arrays.asList(L, R));
  }

  public static ArrayList<Integer> flipN2(String A) {
    if (A == null || A.length() == 0) {
      return new ArrayList<>();
    }
    int l = A.length();
    int L = -1, R = -1, maxSum = -1, curL, curR;
    for (int i = 0; i < l; i++) {
      int curSum = 0;
      curL = i;
      for (int j = i; j < l; j++) {
        int c = (A.charAt(j) == '1') ? -1 : 1;
        curSum += c;
        curR = j;
        if (curSum > maxSum) {
          R = curR;
          maxSum = curSum;
          L = curL;
        }
      }

    }
    if (maxSum == -1) {
      return new ArrayList<>();
    }
    return new ArrayList<>(Arrays.asList(L, R));
  }

  static class IntegerPair {
    Integer first, second;

    public IntegerPair(int first, int second) {
      this.first = first;
      this.second = second;
    }
  }

}